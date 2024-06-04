# Copyright 2024 Grabtaxi Holdings Pte Ltd (GRAB), All rights reserved.
# Use of this source code is governed by an MIT-style license that can be found in the LICENSE file

import torch
import torch
import torch_geometric
from torch_geometric.data import HeteroData


def standardize(x: torch.Tensor) -> torch.Tensor:
    mu = x.mean(dim=0, keepdim=True)
    sigm = x.std(dim=0, keepdim=True)
    z = (x - mu) / sigm
    return z


def standardize_features(graph: HeteroData):
    for nt in graph.node_types:
        graph[nt].x = standardize(graph[nt].x)
    for et in graph.edge_types:
        graph[et].edge_attr = standardize(graph[et].edge_attr)

    return graph
