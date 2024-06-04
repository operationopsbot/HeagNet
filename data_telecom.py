# Copyright 2024 Grabtaxi Holdings Pte Ltd (GRAB), All rights reserved.
# Use of this source code is governed by an MIT-style license that can be found in the LICENSE file

import torch
import torch_geometric
from torch_geometric.data import HeteroData
from anomaly_insert import inject_random_block_anomaly


import pandas as pd
import numpy as np


# %% conf

node_types = ["user", "app", "cell", "package"]
edge_types = [
    ("user", "buy", "package"),
    ("user", "live", "cell"),
    ("user", "use", "app"),
]


def create_graph():
    ## construct graph
    print("constructing graph...")
    data = HeteroData()

    # for nt in node_types:
    #     print(nt)
    #     df = pd.read_csv(f"data/telecom/telecom-graph/node_{nt}.txt", header=None)
    #     x = torch.tensor(df.iloc[:, 1:].values).float()
    #     data[nt].x = x

    #     print(f"num_nodes: {data[nt].num_nodes}")

    for et in edge_types:
        print(et)
        dfe = pd.read_csv(
            f"data/telecom/telecom-graph/edge_{et[0]}_{et[1]}_{et[2]}.txt", header=None
        )
        # take the mean over multi-edges
        dfe = dfe.groupby([0, 1]).mean().reset_index()

        row = torch.tensor(dfe[0].values)
        col = torch.tensor(dfe[1].values)

        # reorder idx
        row_col = row * col.max() + col
        sid = torch.argsort(row_col)

        row = row[sid]
        col = col[sid]

        # put it in graph

        edge_index = torch.stack([row, col])
        data[et].edge_index = edge_index

        xe = torch.tensor(dfe.iloc[:, 2:].values).float()
        xe = xe[sid]
        data[et].edge_attr = xe

        print(f"num_edges: {data[et].num_edges}")

    # store graph
    print("storing graph...")
    torch.save(data, "storage/telecom-graph.pt")
    print("DONE")


def synth_random_anomalies():
    # generate nd store data
    import argparse

    parser = argparse.ArgumentParser(description="Hetero_GraphBEAN")
    parser.add_argument("--name", type=str, default="telecom-anomaly", help="name")
    parser.add_argument("--n-graph", type=int, default=1, help="n graph")

    args = vars(parser.parse_args())

    graph = torch.load(f"storage/telecom-graph.pt")
    print(graph)

    all_nodes = {nt: graph[nt].num_nodes for nt in graph.node_types}
    sum_all_nodes = sum([graph[nt].num_nodes for nt in graph.node_types])
    sum_all_edges = sum([graph[et].num_edges for et in graph.edge_types])

    graph_anomaly_list = []
    for i in range(args["n_graph"]):
        print(f"GRAPH ANOMALY {i} >>>>>>>>>>>>>>")
        print(f"all: nodes = {sum_all_nodes}, edges = {sum_all_edges} | {all_nodes}")

        num_group = 15
        num_nodes_range_dict = {
            ("user", "buy", "package"): ((6, 20), (1, 3)),
            ("user", "live", "cell"): ((6, 20), (6, 30)),
            ("user", "use", "app"): ((4, 10), (2, 6)),
        }

        # num_group = 15
        # num_nodes_range_dict = {
        #     ("user", "buy", "package"): ((1, 4), (1, 3)),
        #     ("user", "live", "cell"): ((1, 4), (6, 20)),
        #     ("user", "use", "app"): ((1, 4), (1, 4)),
        # }

        graph_multi_dense = inject_random_block_anomaly(
            graph, num_group=num_group, num_nodes_range_dict=num_nodes_range_dict
        )
        graph_anomaly_list.append(graph_multi_dense)

        print("\nAnomalies -->")
        for nt in graph_multi_dense.node_types:
            print(
                f"{nt}: {graph_multi_dense[nt].y.sum()}/{graph_multi_dense[nt].y.shape[0]}  ({graph_multi_dense[nt].y.sum()/graph_multi_dense[nt].y.shape[0]})"
            )
        for et in graph_multi_dense.edge_types:
            print(
                f"{et}: {graph_multi_dense[et].ye.sum()}/{graph_multi_dense[et].ye.shape[0]}  ({graph_multi_dense[et].ye.sum()/graph_multi_dense[et].ye.shape[0]})"
            )
        print("\n")

        print()

    dataset = {"args": args, "graph": graph, "graph_anomaly_list": graph_anomaly_list}

    torch.save(dataset, f"storage/{args['name']}.pt")


if __name__ == "__main__":
    # create_graph()
    synth_random_anomalies()
