# Copyright 2024 Grabtaxi Holdings Pte Ltd (GRAB), All rights reserved.
# Use of this source code is governed by an MIT-style license that can be found in the LICENSE file

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_sparse import SparseTensor
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader, NeighborLoader, LinkNeighborLoader
import torch_geometric.transforms as T


from torch_geometric.typing import Adj, PairTensor, OptTensor
from typing import List, Optional, Tuple, Union, Dict

from models.hetero_conv import HeagNetConv
import time
from tqdm import tqdm
import copy

from torch_geometric.typing import (
    PairTensor,
    OptTensor,
    NodeType,
    EdgeType,
    Adj,
    Metadata,
)

from models.sampler import EdgePredictionSampler
from utils.combine import combine_rev_edges


def make_dict(
    x: Union[int, Dict[NodeType, int], Dict[EdgeType, int]], keys: Optional[List]
):
    if isinstance(x, int):
        return {k: x for k in keys}
    else:
        return x


def relu_dropout(
    x: Tensor,
    dropout_prob: float,
    training: bool,
) -> Tensor:
    x = F.relu(x)
    # x = F.leaky_relu(x)
    if dropout_prob > 0.0:
        x = F.dropout(x, p=dropout_prob, training=training)
    return x


def apply_relu_dropout(
    x_dict: Union[Dict[NodeType, Tensor], Dict[EdgeType, Tensor]],
    dropout_prob: float,
    training: bool,
) -> Tensor:
    for k, x in x_dict.items():
        x = F.relu(x)
        # x = F.leaky_relu(x)
        if dropout_prob > 0.0:
            x = F.dropout(x, p=dropout_prob, training=training)
        x_dict[k] = x
    return x_dict


def is_sorted(x: Tensor):
    return torch.all(x[1:] >= x[:-1])


def search_in_edge_index(
    edge_index: Tensor,
    row_arr: Tensor,
    col_arr: Tensor,
    n_node_row: int,
    n_node_col: int,
):
    # searching
    if is_sorted(edge_index[0]):
        row_col_orig = edge_index[0] * n_node_col + edge_index[1]
        row_col_batch = row_arr * n_node_col + col_arr
        id_searched = torch.searchsorted(row_col_orig, row_col_batch)
    elif is_sorted(edge_index[1]):
        row_col_orig = edge_index[1] * n_node_row + edge_index[0]
        row_col_batch = col_arr * n_node_row + row_arr
        id_searched = torch.searchsorted(row_col_orig, row_col_batch)
    else:
        row_col_orig = edge_index[0] * n_node_col + edge_index[1]
        row_col_batch = row_arr * n_node_col + col_arr
        id_searched = torch.where(torch.eq(row_col_orig[:, None], row_col_batch))[0]

    return id_searched


class HeagNet(nn.Module):
    def __init__(
        self,
        meta: Metadata,
        in_channels_node: Union[int, Dict[NodeType, int]],
        in_channels_edge: Union[int, Dict[EdgeType, int]],
        hidden_channels_node: Union[int, Dict[NodeType, int]] = 32,
        hidden_channels_edge: Union[int, Dict[EdgeType, int]] = 32,
        latent_channels_node: Union[int, Dict[NodeType, int]] = 64,
        edge_pred_latent: Union[int, Dict[EdgeType, int]] = 64,
        n_layers_encoder: int = 4,
        n_layers_decoder: int = 4,
        n_layers_mlp: int = 4,
        dropout_prob: float = 0.0,
        bidirectional: bool = False,
    ):
        super().__init__()

        self.meta = meta
        self.node_types = meta[0]
        self.edge_types = meta[1]

        self.in_channels_node = make_dict(in_channels_node, self.node_types)
        self.in_channels_edge = make_dict(in_channels_edge, self.edge_types)
        self.hidden_channels_node = make_dict(hidden_channels_node, self.node_types)
        self.hidden_channels_edge = make_dict(hidden_channels_edge, self.edge_types)
        self.latent_channels_node = make_dict(latent_channels_node, self.node_types)
        self.edge_pred_latent = make_dict(edge_pred_latent, self.edge_types)
        self.n_layers_encoder = n_layers_encoder
        self.n_layers_decoder = n_layers_decoder
        self.n_layers_mlp = n_layers_mlp
        self.dropout_prob = dropout_prob
        self.bidirectional = bidirectional

        self.create_encoder()
        self.create_feature_decoder()
        self.create_structure_decoder()

    def create_encoder(self):
        self.encoder_convs = nn.ModuleList()
        for i in range(self.n_layers_encoder):
            if i == 0 and i == self.n_layers_encoder - 1:
                in_channels_node = self.in_channels_node
                out_channels_node = self.latent_channels_node
                in_channels_edge = self.in_channels_edge
                out_channels_edge = None
            elif i == 0:
                in_channels_node = self.in_channels_node
                out_channels_node = self.hidden_channels_node
                in_channels_edge = self.in_channels_edge
                out_channels_edge = self.hidden_channels_edge
            elif i == self.n_layers_encoder - 1:
                in_channels_node = self.hidden_channels_node
                out_channels_node = self.latent_channels_node
                in_channels_edge = self.hidden_channels_edge
                out_channels_edge = None
            else:
                in_channels_node = self.hidden_channels_node
                out_channels_node = self.hidden_channels_node
                in_channels_edge = self.hidden_channels_edge
                out_channels_edge = self.hidden_channels_edge

            if i == self.n_layers_encoder - 1:
                node_self_loop = False
            else:
                node_self_loop = True

            self.encoder_convs.append(
                HeagNetConv(
                    self.meta,
                    in_channels_node=in_channels_node,
                    out_channels_node=out_channels_node,
                    in_channels_edge=in_channels_edge,
                    out_channels_edge=out_channels_edge,
                    node_self_loop=node_self_loop,
                    bidirectional=self.bidirectional,
                )
            )

    def create_feature_decoder(self):
        self.decoder_convs = nn.ModuleList()
        for i in range(self.n_layers_decoder):
            if i == 0 and i == self.n_layers_decoder - 1:
                in_channels_node = self.latent_channels_node
                out_channels_node = self.in_channels_node
                in_channels_edge = None
                out_channels_edge = self.in_channels_edge
            elif i == 0:
                in_channels_node = self.latent_channels_node
                out_channels_node = self.hidden_channels_node
                in_channels_edge = None
                out_channels_edge = self.hidden_channels_edge
            elif i == self.n_layers_decoder - 1:
                in_channels_node = self.hidden_channels_node
                out_channels_node = self.in_channels_node
                in_channels_edge = self.hidden_channels_edge
                out_channels_edge = self.in_channels_edge
            else:
                in_channels_node = self.hidden_channels_node
                out_channels_node = self.hidden_channels_node
                in_channels_edge = self.hidden_channels_edge
                out_channels_edge = self.hidden_channels_edge

            self.decoder_convs.append(
                HeagNetConv(
                    self.meta,
                    in_channels_node=in_channels_node,
                    out_channels_node=out_channels_node,
                    in_channels_edge=in_channels_edge,
                    out_channels_edge=out_channels_edge,
                    bidirectional=self.bidirectional,
                )
            )

    def create_structure_decoder(self):
        self.u_mlp = nn.ModuleDict()
        self.v_mlp = nn.ModuleDict()

        for et in self.edge_types:
            src, rel, dst = et
            str_edge_type = "__".join(et)

            u_mlp_layers = nn.ModuleList()
            v_mlp_layers = nn.ModuleList()

            for i in range(self.n_layers_mlp):
                if i == 0:
                    in_channels_u = self.latent_channels_node[src]
                    in_channels_v = self.latent_channels_node[dst]
                else:
                    in_channels_u = self.edge_pred_latent[et]
                    in_channels_v = self.edge_pred_latent[et]

                out_channels_u = self.edge_pred_latent[et]
                out_channels_v = self.edge_pred_latent[et]

                u_mlp_layers.append(Linear(in_channels_u, out_channels_u))
                v_mlp_layers.append(Linear(in_channels_v, out_channels_v))

            self.u_mlp[str_edge_type] = u_mlp_layers
            self.v_mlp[str_edge_type] = v_mlp_layers

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        xe_dict: Dict[EdgeType, Tensor],
        adj_dict: Dict[EdgeType, Adj],
        edge_pred_samples_dict: Dict[EdgeType, SparseTensor],
    ) -> Dict[str, Tensor]:
        # print("--conv encoder forward", end=" : ")
        # t0 = time.time()

        ## encoder
        for i, conv in enumerate(self.encoder_convs):
            x_dict, xe_dict = conv(x_dict, adj_dict, xe_dict=xe_dict)
            if i != self.n_layers_encoder - 1:
                x_dict = apply_relu_dropout(x_dict, self.dropout_prob, self.training)
                xe_dict = apply_relu_dropout(xe_dict, self.dropout_prob, self.training)

        # print(f"{time.time()-t0:.4f} second")

        # print("--conv decoder forward", end=" : ")
        # t0 = time.time()

        ## get latent vars
        z_dict = x_dict

        ## feature decoder
        for i, conv in enumerate(self.decoder_convs):
            x_dict, xe_dict = conv(x_dict, adj_dict, xe_dict=xe_dict)
            if i != self.n_layers_decoder - 1:
                x_dict = apply_relu_dropout(x_dict, self.dropout_prob, self.training)
                xe_dict = apply_relu_dropout(xe_dict, self.dropout_prob, self.training)

        # print(f"{time.time()-t0:.4f} second")

        # print("--feature decoder forward", end=" : ")
        # t0 = time.time()

        ## structure decoder
        zu_dict = {}
        zv_dict = {}
        for et in adj_dict.keys():
            src, rel, dst = et
            zu_dict[et] = z_dict[src]
            zv_dict[et] = z_dict[dst]

        for i in range(self.n_layers_mlp):
            for et in adj_dict.keys():
                src, rel, dst = et
                str_edge_type = "__".join(et)

                layer = self.u_mlp[str_edge_type][i]
                zu_dict[et] = layer(zu_dict[et])

                layer = self.v_mlp[str_edge_type][i]
                zv_dict[et] = layer(zv_dict[et])

            if i != self.n_layers_mlp - 1:
                zu_dict = apply_relu_dropout(zu_dict, self.dropout_prob, self.training)
                zv_dict = apply_relu_dropout(zv_dict, self.dropout_prob, self.training)

        # print(f"{time.time()-t0:.4f} second")

        # print("--edge pred forward", end=" : ")
        # t0 = time.time()

        eprob_dict = {}
        # for et in self.edge_types:
        for et in edge_pred_samples_dict.keys():
            src, rel, dst = et

            zu = zu_dict[et]
            zv = zv_dict[et]
            edge_pred_samples = edge_pred_samples_dict[et]

            zu_edge = zu[edge_pred_samples.storage.row()]
            zv_edge = zv[edge_pred_samples.storage.col()]

            eprob = torch.sigmoid(torch.sum(zu_edge * zv_edge, dim=1))
            eprob_dict[et] = eprob

        # print(f"{time.time()-t0:.4f} second")

        # collect results
        result = {
            "x_dict": x_dict,
            "xe_dict": xe_dict,
            "z_dict": z_dict,
            "eprob_dict": eprob_dict,
        }

        return result

    ## FOR INFERENCE
    def apply_conv(
        self,
        conv: torch.nn.Module,
        batch: HeteroData,
        activation: bool = True,
    ):
        # get data
        # need to do this, as the batch sampler were ordered by the the 2nd node type, instead of the first
        x_dict = {k: batch[k].x for k in batch.node_types}

        adj_dict = {}
        xe_dict = {}
        eid_dict = {}
        for k in batch.edge_types:
            if batch[k].num_edges > 0:
                xe = batch[k].edge_attr
                eid = torch.arange(xe.shape[0], device=xe.device)
                edge_attr = torch.cat([xe, eid[:, None]], dim=1)

                spt = SparseTensor.from_edge_index(
                    batch[k].edge_index,
                    sparse_sizes=(batch[k[0]].num_nodes, batch[k[2]].num_nodes),
                    edge_attr=edge_attr,
                )

                adj_dict[k] = spt.set_value(None)
                value_new = spt.storage.value()
                xe_dict[k] = value_new[:, :-1]
                eid_dict[k] = value_new[:, -1].long()

        x_dict, xe_dict = conv(x_dict, adj_dict, xe_dict=xe_dict)
        if activation:
            x_dict = apply_relu_dropout(x_dict, self.dropout_prob, self.training)
            xe_dict = apply_relu_dropout(xe_dict, self.dropout_prob, self.training)

        # reorder back
        if xe_dict is not None:
            for k in xe_dict.keys():
                xe2 = torch.zeros_like(xe_dict[k])
                xe2[eid_dict[k]] = xe_dict[k]
                xe_dict[k] = xe2

        return x_dict, xe_dict

    def new_graph_data(
        self, data: HeteroData, x_dict: Dict[str, Tensor], xe_dict: Dict[str, Tensor]
    ):
        graph = HeteroData()

        for nt in data.node_types:
            graph[nt].x = x_dict[nt]

        for et in data.edge_types:
            graph[et].edge_index = data[et].edge_index
            if xe_dict is not None:
                graph[et].edge_attr = xe_dict[et]
            else:
                graph[et].edge_attr = torch.zeros(graph[et].num_edges, 0)

        return graph

    def inference(
        self,
        dataset: HeteroData,
        node_batch_size: Dict[NodeType, int],
        mlp_batch_size: Dict[NodeType, int],
        epred_batch_size: Dict[EdgeType, int],
        device,
        edge_pred_samples_dict: Optional[Dict[EdgeType, SparseTensor]] = None,
        edge_pred_mult: int = 2,
        progress_bar: bool = True,
        **kwargs,
    ) -> Dict[str, Tensor]:
        ntypes, etypes = dataset.metadata()

        graph = copy.copy(dataset)

        transform = T.ToUndirected()
        graph_sampling = transform(copy.copy(graph))

        kwargs["shuffle"] = False

        # progress bar
        total_iter = (
            (len(ntypes)) * (self.n_layers_encoder + self.n_layers_decoder)
            + (len(etypes)) * self.n_layers_mlp
            + len(etypes)
        )
        if progress_bar:
            pbar = tqdm(total=total_iter, leave=False)
            pbar.set_description(f"Evaluation")

        # init
        x_dict = {}
        xe_dict = {}
        z_dict = {}

        ## encoder & decoder
        for i, conv in enumerate(self.encoder_convs + self.decoder_convs):
            # print(f"conv layer {i}...", flush=True)
            # loaders
            node_loaders = {
                key: NeighborLoader(
                    graph_sampling,
                    num_neighbors={
                        k: [-1] if k[2] == key else [0]
                        for k in graph_sampling.edge_types
                    },
                    batch_size=node_batch_size[key],
                    input_nodes=key,
                    **kwargs,
                )
                for key in ntypes
            }

            ## next nodes
            x_dict = {}
            xe_dict = {}
            xe_rev_dict = {}
            for nt in ntypes:
                x_list = []
                xe_next_dict = {}
                row_next_dict = {}
                col_next_dict = {}

                xe_rev_next_dict = {}
                row_rev_next_dict = {}
                col_rev_next_dict = {}

                for et in etypes:
                    if et[2] == nt:
                        xe_next_dict[et] = []
                        row_next_dict[et] = []
                        col_next_dict[et] = []
                    elif et[0] == nt:
                        xe_rev_next_dict[et] = []
                        row_rev_next_dict[et] = []
                        col_rev_next_dict[et] = []

                for batch in node_loaders[nt]:
                    batch = combine_rev_edges(batch)
                    batch = batch.to(device)
                    actv = (
                        False
                        if (i == self.n_layers_encoder - 1)
                        or (i == self.n_layers_encoder + self.n_layers_decoder - 1)
                        else True
                    )

                    x_out_dict, xe_out_dict = self.apply_conv(conv, batch, actv)
                    x_list.append(x_out_dict[nt].cpu())

                    if i != self.n_layers_encoder - 1:
                        for et in etypes:
                            if et[2] == nt and batch[et].num_edges > 0:
                                xe_next_dict[et].append(xe_out_dict[et].cpu())

                                row_nid = batch[et[0]]["n_id"][batch[et].edge_index[0]]
                                col_nid = batch[et[2]]["n_id"][batch[et].edge_index[1]]
                                row_next_dict[et].append(row_nid.cpu())
                                col_next_dict[et].append(col_nid.cpu())
                            elif et[0] == nt and batch[et].num_edges > 0:
                                xe_rev_next_dict[et].append(xe_out_dict[et].cpu())

                                row_nid = batch[et[0]]["n_id"][batch[et].edge_index[0]]
                                col_nid = batch[et[2]]["n_id"][batch[et].edge_index[1]]
                                row_rev_next_dict[et].append(row_nid.cpu())
                                col_rev_next_dict[et].append(col_nid.cpu())

                x_dict[nt] = torch.cat(x_list, dim=0)

                if i != self.n_layers_encoder - 1:
                    for et in etypes:
                        if et[2] == nt:
                            xe_cmb = torch.cat(xe_next_dict[et], dim=0)
                            row_cmb = torch.cat(row_next_dict[et], dim=0)
                            col_cmb = torch.cat(col_next_dict[et], dim=0)

                            # searching
                            id_in_orig = search_in_edge_index(
                                edge_index=graph[et].edge_index,
                                row_arr=row_cmb,
                                col_arr=col_cmb,
                                n_node_row=graph[et[0]].num_nodes,
                                n_node_col=graph[et[2]].num_nodes,
                            )

                            xe = torch.zeros((graph[et].num_edges, xe_cmb.shape[1]))
                            xe[id_in_orig] = xe_cmb

                            xe_dict[et] = xe

                        elif et[0] == nt:
                            xe_cmb = torch.cat(xe_rev_next_dict[et], dim=0)
                            row_cmb = torch.cat(row_rev_next_dict[et], dim=0)
                            col_cmb = torch.cat(col_rev_next_dict[et], dim=0)

                            # searching
                            id_in_orig = search_in_edge_index(
                                edge_index=graph[et].edge_index,
                                row_arr=row_cmb,
                                col_arr=col_cmb,
                                n_node_row=graph[et[0]].num_nodes,
                                n_node_col=graph[et[2]].num_nodes,
                            )

                            xe = torch.zeros((graph[et].num_edges, xe_cmb.shape[1]))
                            xe[id_in_orig] = xe_cmb

                            xe_rev_dict[et] = xe
                else:
                    xe_dict = None

                if progress_bar:
                    pbar.update(1)

            # update graph
            if i < self.n_layers_encoder + self.n_layers_decoder - 1:
                graph = self.new_graph_data(graph, x_dict, xe_dict)
                graph_sampling = transform(copy.copy(graph))

            # store latent
            if i == self.n_layers_encoder - 1:
                z_dict = x_dict

        ## save some memory
        ## as graph conv is no longer needed
        # print(f"convolution layers completed. deleting graph", flush=True)
        del graph
        del graph_sampling

        ## structure decoder

        # epred sampler and loaders
        # print(f"loader constructions...", flush=True)
        # loaders
        mlp_loaders = {
            key: torch.utils.data.DataLoader(
                torch.arange(dataset[key].num_nodes),
                batch_size=mlp_batch_size[key],
                **kwargs,
            )
            for key in ntypes
        }

        # print(f"start mlp training...", flush=True)
        # edge prediction
        eprob_dict = {}
        edge_pred_samples_dict = {}
        # for et in self.edge_types:
        for et in etypes:
            # print(f"edge pred {et}...", flush=True)
            src, rel, dst = et
            str_edge_type = "__".join(et)

            # u and v nodes
            zu = z_dict[src]
            zv = z_dict[dst]

            for i in range(self.n_layers_mlp):
                # print(f"  mlp layer {i}", flush=True)

                # u nodes
                zu_list = []
                layer = self.u_mlp[str_edge_type][i]
                for batch in mlp_loaders[src]:
                    out = layer(zu[batch].to(device))
                    if i != self.n_layers_mlp - 1:
                        out = relu_dropout(out, self.dropout_prob, self.training)
                    zu_list.append(out.cpu())
                zu = torch.cat(zu_list, dim=0)

                # v nodes
                zv_list = []
                layer = self.v_mlp[str_edge_type][i]
                for batch in mlp_loaders[dst]:
                    out = layer(zv[batch].to(device))
                    if i != self.n_layers_mlp - 1:
                        out = relu_dropout(out, self.dropout_prob, self.training)
                    zv_list.append(out.cpu())
                zv = torch.cat(zv_list, dim=0)

                if progress_bar:
                    pbar.update(1)

            # epred loaders
            adj = SparseTensor.from_edge_index(
                dataset[et].edge_index,
                sparse_sizes=(dataset[src].num_nodes, dataset[dst].num_nodes),
            )

            edge_pred_sampler = EdgePredictionSampler(adj, mult=edge_pred_mult)
            edge_pred_samples = edge_pred_sampler.sample()

            edge_pred_samples_dict[et] = edge_pred_samples

            epred_loader = torch.utils.data.DataLoader(
                torch.arange(edge_pred_samples.nnz()),
                batch_size=epred_batch_size[et],
                **kwargs,
            )

            eprob_list = []
            for batch in epred_loader:
                zu_edge = zu[edge_pred_samples.storage.row()[batch]].to(device)
                zv_edge = zv[edge_pred_samples.storage.col()[batch]].to(device)

                eprob = torch.sigmoid(torch.sum(zu_edge * zv_edge, dim=1))
                eprob_list.append(eprob.cpu())

            eprob_dict[et] = torch.cat(eprob_list, dim=0)

            if progress_bar:
                pbar.update(1)

        # collect results
        result = {
            "x_dict": x_dict,
            "xe_dict": xe_dict,
            "z_dict": z_dict,
            "eprob_dict": eprob_dict,
            "edge_pred_samples_dict": edge_pred_samples_dict,
        }

        return result
