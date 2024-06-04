# Copyright 2024 Grabtaxi Holdings Pte Ltd (GRAB), All rights reserved.
# Use of this source code is governed by an MIT-style license that can be found in the LICENSE file

from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import torch

from torch import Tensor
import torch.nn as nn
from torch_sparse import SparseTensor, matmul
from torch_scatter import scatter
import torch.nn.functional as F

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.conv.hgt_conv import group


from torch_geometric.typing import (
    PairTensor,
    OptTensor,
    NodeType,
    EdgeType,
    Adj,
    Metadata,
)


class HeagNetConv(torch.nn.Module):
    def __init__(
        self,
        meta: Metadata,
        in_channels_node: Dict[NodeType, int],
        out_channels_node: Dict[NodeType, int],
        in_channels_edge: Optional[Dict[EdgeType, int]] = None,
        out_channels_edge: Optional[Dict[EdgeType, int]] = None,
        bidirectional: bool = False,
        node_self_loop: bool = True,
        normalize: bool = True,
        bias: bool = True,
        agg_inner: List[str] = ["mean", "max"],
        agg_inter: str = "mean",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.node_types = meta[0]
        self.edge_types = meta[1]

        self.in_channels_node = in_channels_node
        self.out_channels_node = out_channels_node
        self.in_channels_edge = in_channels_edge
        self.out_channels_edge = out_channels_edge

        self.bidirectional = bidirectional
        self.node_self_loop = node_self_loop
        self.normalize = normalize

        self.agg_inner = agg_inner
        self.agg_inter = agg_inter

        self.input_has_edge_channel = in_channels_edge is not None
        self.output_has_edge_channel = out_channels_edge is not None

        # convolutions
        node_convs = {}
        edge_convs = {}
        for et in self.edge_types:
            src, rel, dst = et
            if self.input_has_edge_channel:
                in_channels = (
                    self.in_channels_node[src],
                    self.in_channels_node[dst],
                    self.in_channels_edge[et],
                )
            else:
                in_channels = (self.in_channels_node[src], self.in_channels_node[dst])

            # node conv
            conv = BEANConvNode(
                in_channels,
                self.out_channels_node[dst],
                normalize=normalize,
                bias=bias,
                agg=agg_inner,
                **kwargs,
            )
            str_edge_type = "__".join(et)
            node_convs[str_edge_type] = conv

            # if the message passing is bidirectional
            ## if it's bidirectional and the node types are the same,
            ## the adjacency matrix is expected to be symmetric.
            if bidirectional and src != dst:
                conv = BEANConvNode(
                    in_channels,
                    self.out_channels_node[src],
                    flows="v->u",
                    normalize=normalize,
                    bias=bias,
                    agg=agg_inner,
                    **kwargs,
                )
                rev_str_edge_type = "__".join([dst, "rev_" + rel, src])
                node_convs[rev_str_edge_type] = conv

            # edge conv
            if self.output_has_edge_channel:
                conv = BEANConvEdge(
                    in_channels,
                    self.out_channels_edge[et],
                    normalize=normalize,
                    bias=bias,
                    **kwargs,
                )
                edge_convs["__".join(et)] = conv

        # convert to PyTorch ModuleDict
        self.node_convs = nn.ModuleDict(node_convs)
        self.edge_convs = nn.ModuleDict(edge_convs)

        # linear component after gathering all msg from different edge types
        node_lins = {}
        for nt in self.node_types:
            if self.node_self_loop:
                lin_channels = in_channels_node[nt] + out_channels_node[nt]
            else:
                lin_channels = out_channels_node[nt]

            node_lins[nt] = Linear(lin_channels, out_channels_node[nt], bias=bias)

        self.node_lins = nn.ModuleDict(node_lins)

    def reset_parameters(self):
        for conv in self.node_convs.values():
            conv.reset_parameters()
        for conv in self.edge_convs.values():
            conv.reset_parameters()
        for lin in self.node_lins.values():
            lin.reset_parameters()

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        adj_dict: Dict[EdgeType, Adj],
        xe_dict: Dict[EdgeType, Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        x_out_list_dict = defaultdict(list)
        xe_out_dict = {} if self.output_has_edge_channel else None

        for et, adj in adj_dict.items():
            src, rel, dst = et
            str_edge_type = "__".join(et)

            ## node convolution
            xe = xe_dict[et] if self.input_has_edge_channel else None  # edge
            conv = self.node_convs[str_edge_type]
            out = conv((x_dict[src], x_dict[dst]), adj, xe=xe)
            x_out_list_dict[dst].append(out)

            if self.bidirectional and src != dst:
                rev_str_edge_type = "__".join([dst, "rev_" + rel, src])
                conv = self.node_convs[rev_str_edge_type]
                out = conv((x_dict[src], x_dict[dst]), adj, xe=xe)
                x_out_list_dict[src].append(out)

            ## edge convolution
            if self.output_has_edge_channel:
                conv = self.edge_convs[str_edge_type]
                out_e = conv((x_dict[src], x_dict[dst]), adj, xe=xe)

                xe_out_dict[et] = out_e

        # stack all messages to node
        x_out_dict = {}
        for nt in x_dict.keys():
            out = group(x_out_list_dict[nt], self.agg_inter)
            if self.node_self_loop:
                if torch.is_tensor(out):
                    out = torch.cat((x_dict[nt], out), dim=1)
                else:
                    out = torch.cat(
                        (
                            x_dict[nt],
                            torch.zeros(
                                x_dict[nt].shape[0],
                                self.out_channels_node[nt],
                                device=x_dict[nt].device,
                            ),
                        ),
                        dim=1,
                    )
            else:
                if not torch.is_tensor(out):
                    out = torch.zeros(
                        x_dict[nt].shape[0],
                        self.out_channels_node[nt],
                        device=x_dict[nt].device,
                    )

            out = self.node_lins[nt](out)
            x_out_dict[nt] = out

        return x_out_dict, xe_out_dict


class BEANConvNode(MessagePassing):
    def __init__(
        self,
        in_channels: Tuple[int, int, Optional[int]],
        out_channels: int,
        flows: str = "u->v",
        normalize: bool = True,
        use_activation: bool = True,
        bias: bool = True,
        agg: List[str] = ["mean", "max"],
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.flows = flows
        self.normalize = normalize
        self.use_activation = use_activation
        self.agg = agg

        self.input_has_edge_channel = len(in_channels) == 3

        n_agg = len(agg)
        # calculate in channels
        if self.input_has_edge_channel:
            if flows == "v->u":
                self.in_channels_all = n_agg * in_channels[1] + n_agg * in_channels[2]
            else:
                self.in_channels_all = n_agg * in_channels[0] + n_agg * in_channels[2]
        else:
            if flows == "v->u":
                self.in_channels_all = n_agg * in_channels[1]
            else:
                self.in_channels_all = n_agg * in_channels[0]

        self.lin = Linear(self.in_channels_all, out_channels, bias=bias)

        if normalize:
            self.bn = nn.BatchNorm1d(out_channels)

        if use_activation:
            self.act = nn.PReLU()

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, x: PairTensor, adj: SparseTensor, xe: OptTensor = None) -> Tensor:
        assert self.input_has_edge_channel == (xe is not None)

        # propagate_type: (x: PairTensor)
        out = self.propagate(adj, x=x, xe=xe)

        # lin layer
        out = self.lin(out)
        if self.normalize:
            # if out.shape[0] < 2:
            # print(out.shape)
            out = self.bn(out)

        if self.use_activation:
            out = self.act(out)

        return out

    def message_and_aggregate(
        self, adj: SparseTensor, x: PairTensor, xe: OptTensor
    ) -> Tensor:
        xu, xv = x
        adj = adj.set_value(None, layout=None)

        ## Node V to node U
        if self.flows == "v->u":
            # messages node to node
            msg_v2u_list = [matmul(adj, xv, reduce=ag) for ag in self.agg]

            # messages edge to node
            if xe is not None:
                msg_e2u_list = [
                    scatter(
                        xe,
                        adj.storage.row(),
                        dim=0,
                        reduce=ag,
                        dim_size=adj.sparse_size(0),
                    )
                    for ag in self.agg
                ]

            # collect all msg
            if xe is not None:
                msg_2u = torch.cat((*msg_v2u_list, *msg_e2u_list), dim=1)
            else:
                msg_2u = torch.cat((*msg_v2u_list,), dim=1)

            return msg_2u

        ## Node U to node V
        else:
            # print(f"matmul : {type(adj)} x {type(xu)}")
            # if not torch.is_tensor(xu):
            #     print("xu : ")
            #     print(xu)
            #     a = 1

            msg_u2v_list = [matmul(adj.t(), xu, reduce=ag) for ag in self.agg]

            # messages edge to node
            if xe is not None:
                msg_e2v_list = [
                    scatter(
                        xe,
                        adj.storage.col(),
                        dim=0,
                        reduce=ag,
                        dim_size=adj.sparse_size(1),
                    )
                    for ag in self.agg
                ]

            # collect all msg (including self loop)
            if xe is not None:
                msg_2v = torch.cat((*msg_u2v_list, *msg_e2v_list), dim=1)
            else:
                msg_2v = torch.cat((*msg_u2v_list,), dim=1)

            return msg_2v


class BEANConvEdge(MessagePassing):
    def __init__(
        self,
        in_channels: Tuple[int, int, Optional[int]],  # node u, node v, edge
        out_channels: int,
        normalize: bool = True,
        use_activation: bool = True,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.use_activation = use_activation

        self.input_has_edge_channel = len(in_channels) == 3

        if self.input_has_edge_channel:
            self.in_channels_e = in_channels[0] + in_channels[1] + in_channels[2]
        else:
            self.in_channels_e = in_channels[0] + in_channels[1]

        self.lin_e = Linear(self.in_channels_e, out_channels, bias=bias)

        if normalize:
            self.bn_e = nn.BatchNorm1d(out_channels)

        if use_activation:
            self.act_e = nn.PReLU()

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_e.reset_parameters()

    def forward(self, x: PairTensor, adj: SparseTensor, xe: Tensor) -> Tensor:
        """"""

        # propagate_type: (x: PairTensor)
        out_e = self.propagate(adj, x=x, xe=xe)

        # lin layer
        out_e = self.lin_e(out_e)

        if self.normalize:
            out_e = self.bn_e(out_e)

        if self.use_activation:
            out_e = self.act_e(out_e)

        return out_e

    def message_and_aggregate(
        self, adj: SparseTensor, x: PairTensor, xe: OptTensor
    ) -> Tensor:
        xu, xv = x
        adj = adj.set_value(None, layout=None)

        # collect all msg (including self loop)
        if xe is not None:
            msg_2e = torch.cat(
                (xe, xu[adj.storage.row()], xv[adj.storage.col()]), dim=1
            )
        else:
            msg_2e = torch.cat((xu[adj.storage.row()], xv[adj.storage.col()]), dim=1)

        return msg_2e
