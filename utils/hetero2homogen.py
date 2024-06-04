# Copyright 2024 Grabtaxi Holdings Pte Ltd (GRAB), All rights reserved.
# Use of this source code is governed by an MIT-style license that can be found in the LICENSE file

import torch
import numpy as np

from torch_geometric.data import HeteroData, Data
from torch_geometric.data.hetero_data import to_homogeneous_edge_index
from sklearn.decomposition import PCA  # to apply PCA

from torch_geometric.typing import (
    NodeType,
    EdgeType,
)
from typing import List, Optional, Tuple, Union, Dict


def to_homogeneous(
    graph: HeteroData,
    node_groups: List[List[NodeType]],
    edge_groups: List[List[EdgeType]],
    use_pca: Optional[bool] = True,
    pca_node_dim: Optional[int] = None,
    pca_edge_dim: Optional[int] = None,
) -> Data:
    # edge index transform
    edge_index, node_slices, edge_slices = to_homogeneous_edge_index(graph)
    device = edge_index.device if edge_index is not None else None

    # column slice
    node_column_slices = {}
    cumsum = 0
    for k, v in node_slices.items():
        if k not in node_column_slices:
            next_cs = cumsum + graph[k].x.shape[1]
            node_column_slices[k] = (cumsum, next_cs)
            for group in node_groups:
                if k in group:
                    for k2 in group:
                        node_column_slices[k2] = (cumsum, next_cs)
            cumsum = next_cs
    node_column_size = cumsum

    # edge column slice
    edge_column_slices = {}
    cumsum = 0
    for k, v in edge_slices.items():
        if k not in edge_column_slices:
            next_cs = cumsum + graph[k].edge_attr.shape[1]
            edge_column_slices[k] = (cumsum, next_cs)
            for group in edge_groups:
                if k in group:
                    for k2 in group:
                        edge_column_slices[k2] = (cumsum, next_cs)
            cumsum = next_cs
    edge_column_size = cumsum

    data = Data(**graph._global_store.to_dict())
    if edge_index is not None:
        data.edge_index = edge_index
    data._node_type_names = list(node_slices.keys())
    data._edge_type_names = list(edge_slices.keys())
    data._node_slices = node_slices
    data._edge_slices = edge_slices

    # Combine node attributes into a single tensor:
    num_nodes = list(node_slices.values())[-1][1]
    x_all = torch.zeros(num_nodes, node_column_size, device=device)
    y_all = torch.zeros(num_nodes, device=device)
    for k, v in node_slices.items():
        c = node_column_slices[k]

        x_all[v[0] : v[1], c[0] : c[1]] = graph[k].x
        y_all[v[0] : v[1]] = graph[k].y

        data["x"] = x_all
        data["y"] = y_all

    data.num_nodes = num_nodes

    # Combine edge attributes into a single tensor:
    num_edges = list(edge_slices.values())[-1][1]
    xe_all = torch.zeros(num_edges, edge_column_size, device=device)
    ye_all = torch.zeros(num_edges, device=device)
    for k, v in edge_slices.items():
        c = edge_column_slices[k]

        xe_all[v[0] : v[1], c[0] : c[1]] = graph[k].edge_attr
        ye_all[v[0] : v[1]] = graph[k].ye

        data["edge_attr"] = xe_all
        data["ye"] = ye_all

    data.num_edges = num_edges

    # reduce dim using PCA
    if use_pca:
        if pca_node_dim is None:
            pca_node_dim = x_all.shape[1] // len(node_slices)

        if pca_edge_dim is None:
            pca_edge_dim = xe_all.shape[1] // len(edge_slices)

        npca = PCA(n_components=pca_node_dim)
        x_pca = npca.fit_transform(x_all.numpy())
        data["x"] = torch.from_numpy(x_pca)

        epca = PCA(n_components=pca_edge_dim)
        xe_pca = epca.fit_transform(xe_all.numpy())
        data["edge_attr"] = torch.from_numpy(xe_pca)

    # add node type
    sizes = [offset[1] - offset[0] for offset in node_slices.values()]
    sizes = torch.tensor(sizes, dtype=torch.long, device=device)
    node_type = torch.arange(len(sizes), device=device)
    data.node_type = node_type.repeat_interleave(sizes)

    # add edge type
    sizes = [offset[1] - offset[0] for offset in edge_slices.values()]
    sizes = torch.tensor(sizes, dtype=torch.long, device=device)
    edge_type = torch.arange(len(sizes), device=device)
    data.edge_type = edge_type.repeat_interleave(sizes)

    return data
