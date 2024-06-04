# Copyright 2024 Grabtaxi Holdings Pte Ltd (GRAB), All rights reserved.
# Use of this source code is governed by an MIT-style license that can be found in the LICENSE file

import torch
import torch_geometric

from torch_geometric.data import HeteroData


def combine_rev_edges(graph: HeteroData):
    new_graph = HeteroData()

    for nt in graph.node_types:
        for key in graph[nt]:
            new_graph[nt][key] = graph[nt][key]

    for et in graph.edge_types:
        src, rel, dst = et
        rev_et = (dst, "rev_" + rel, src)

        if (not rel.startswith("rev_")) and (rev_et in graph.edge_types):
            # get indices
            ids = graph[et].edge_index
            rev_edge_index = graph[rev_et].edge_index
            ids_rev = torch.stack([rev_edge_index[1], rev_edge_index[0]])
            ids_cmb = torch.cat([ids, ids_rev], dim=1)

            ids_all, inv, count = ids_cmb.unique(
                dim=1, return_counts=True, return_inverse=True
            )

            ids_duplicate = ids_all[:, count > 1]
            ids_2, count_2 = torch.cat([ids_rev, ids_duplicate], dim=1).unique(
                dim=1, return_counts=True
            )
            flag = count_2 == 1
            ids_additional = ids_rev[:, flag]

            new_graph[et].edge_index = torch.cat([ids, ids_additional], dim=1)
            for key in graph[et]:
                if key != "edge_index" and isinstance(graph[et][key], torch.Tensor):
                    add_val = graph[rev_et][key][flag]
                    new_graph[et][key] = torch.cat([graph[et][key], add_val], dim=0)

        elif not rel.startswith("rev_"):
            for key in graph[et]:
                new_graph[et][key] = graph[et][key]

    return new_graph
