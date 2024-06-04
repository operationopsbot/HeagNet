# Copyright 2024 Grabtaxi Holdings Pte Ltd (GRAB), All rights reserved.
# Use of this source code is governed by an MIT-style license that can be found in the LICENSE file

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import scipy
import math
import random

from torch_sparse import SparseTensor
from typing import List, NamedTuple, Optional, Tuple, Union, Dict

from utils.sparse_combine import spadd
from utils.sprand import sprand

from torch_geometric.data import HeteroData


class EdgePredictionSampler:
    def __init__(
        self,
        adj: SparseTensor,
        n_random: Optional[int] = None,
        mult: Optional[float] = 2.0,
    ):
        self.adj = adj

        if n_random is None:
            n_pos = adj.nnz()
            n_random = int(mult * n_pos)

        self.adj = adj
        self.n_random = n_random

    def sample(self):
        rnd_samples = sprand(self.adj.sparse_sizes(), self.n_random)
        rnd_samples.fill_value_(-1)
        rnd_samples = rnd_samples.to(self.adj.device())

        pos_samples = self.adj.fill_value(2)

        samples = spadd(rnd_samples, pos_samples)
        samples.set_value_(
            torch.minimum(
                samples.storage.value(), torch.ones_like(samples.storage.value())
            ),
            layout="coo",
        )

        return samples


def annotate_target(batch: HeteroData):
    base_node, base_batch_size = list(batch.collect("batch_size").items())[0]

    # initialize
    for nt in batch.node_types:
        batch[nt].target_node = torch.zeros(
            batch[nt].num_nodes, dtype=torch.bool, device=batch[nt].x.device
        )
    for et in batch.edge_types:
        batch[et].target_edge = torch.zeros(
            batch[et].num_edges, dtype=torch.bool, device=batch[et].edge_index.device
        )

    batch[base_node].target_node[:base_batch_size] = True

    for et in batch.edge_types:
        src, rel, dst = et

        if dst == base_node:
            mask = batch[et].edge_index[1] < base_batch_size
            id_src = batch[et].edge_index[0][mask].unique()

            batch[et].target_edge[mask] = True
            batch[src].target_node[id_src] = True

        if src == base_node:
            mask = batch[et].edge_index[0] < base_batch_size
            id_dst = batch[et].edge_index[1][mask].unique()

            batch[et].target_edge[mask] = True
            batch[dst].target_node[id_dst] = True

    return batch


def annotate_edge_pred(
    edge_pred_samples_dict: Dict[str, SparseTensor], annotated_batch: HeteroData
):
    edge_pred_target_mask_dict = {}
    for et, adj in edge_pred_samples_dict.items():
        src, rel, dst = et

        row_in_target = torch.isin(
            adj.storage.row(), annotated_batch[src].target_node.nonzero().squeeze()
        )
        col_in_target = torch.isin(
            adj.storage.col(), annotated_batch[dst].target_node.nonzero().squeeze()
        )
        edge_pred_target_mask_dict[et] = row_in_target | col_in_target

    return edge_pred_target_mask_dict
