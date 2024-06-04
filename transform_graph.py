# Copyright 2024 Grabtaxi Holdings Pte Ltd (GRAB), All rights reserved.
# Use of this source code is governed by an MIT-style license that can be found in the LICENSE file

import torch
from torch_geometric.nn.conv import MessagePassing
from torch import ModuleDict, Tensor
import torch.nn as nn
from tqdm import tqdm
import time
import os

from torch_geometric.nn.conv import HeteroConv, GCNConv, SAGEConv, GATConv
from torch_geometric.datasets.fake import FakeHeteroDataset
from torch_sparse import SparseTensor
from torch_geometric.loader import DataLoader, NeighborLoader
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData

from torch_geometric.data.hetero_data import to_homogeneous_edge_index

from utils.hetero2homogen import to_homogeneous

# name = "telecom-smaller"
# name = "telecom-large"

# name = "reddit"
# name = "gowalla"
name = "brightkite"


nvar_pca = None

print(name)

storage = torch.load(f"storage/{name}-anomaly.pt")
graph = storage["graph_anomaly_list"][0]

# node_groups = [["user"], ["package"], ["cell"], ["app"]]
# edge_groups = [
#     [("user", "buy", "package")],
#     [("user", "live", "cell")],
#     [("user", "use", "app")],
# ]

# node_groups = [["user0"], ["user1"], ["sub0"], ["sub1"]]
# edge_groups = [
#     [("user0", "submission", "sub0")],
#     [("user1", "submission", "sub0")],
#     [("user0", "submission", "sub1")],
#     [("user1", "submission", "sub1")],
# ]

node_groups = [["user"], ["loc0"], ["loc1"], ["loc2"], ["loc3"]]
edge_groups = [
    [("user", "checkin", "loc0")],
    [("user", "checkin", "loc1")],
    [("user", "checkin", "loc2")],
    [("user", "checkin", "loc3")],
]


new_storage = {"args": storage["args"]}

new_graph = to_homogeneous(
    graph,
    node_groups,
    edge_groups,
    use_pca=True,
    pca_node_dim=nvar_pca,
    pca_edge_dim=nvar_pca,
)
new_storage["graph"] = new_graph

graph_anomaly_list = []
for i, gr in enumerate(storage["graph_anomaly_list"]):
    print(i)
    new_gr = to_homogeneous(gr, node_groups, edge_groups)
    graph_anomaly_list.append(new_gr)

new_storage["graph_anomaly_list"] = graph_anomaly_list

torch.save(new_storage, f"storage/{name}-anomaly-homogeneous.pt")
