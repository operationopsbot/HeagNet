# Copyright 2024 Grabtaxi Holdings Pte Ltd (GRAB), All rights reserved.
# Use of this source code is governed by an MIT-style license that can be found in the LICENSE file

import torch
import torch_geometric
from torch_geometric.data import HeteroData
from anomaly_insert import inject_random_block_anomaly


import pandas as pd
import numpy as np


# %% sampling


def reverese_map(x):
    z = np.array([-1] * (x.max() + 1))
    z[x] = np.arange(x.size)

    return z


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

    ## user nodes
    df_user = pd.read_csv(f"data/telecom/telecom-graph/node_user.txt", header=None)

    n_user_sample = 10000
    user_ids = np.random.choice(df_user.shape[0], n_user_sample, replace=False)

    df_user_selected = df_user.iloc[user_ids, :]
    data["user"].x = torch.from_numpy(
        df_user_selected.iloc[:, 1:].to_numpy().astype(np.float32)
    )

    print(f"num_user_nodes: {data['user'].num_nodes} from {df_user.shape[0]}")

    ## edges and other nodes
    for et in edge_types:
        print(et)
        dfe = pd.read_csv(
            f"data/telecom/telecom-graph/edge_{et[0]}_{et[1]}_{et[2]}.txt", header=None
        )

        dfe_selected = dfe[dfe[0].isin(user_ids)]
        other_ids = dfe_selected[1].unique()

        user_rev_ids = reverese_map(user_ids)
        other_rev_ids = reverese_map(other_ids)

        # other node
        df_other = pd.read_csv(
            f"data/telecom/telecom-graph/node_{et[2]}.txt", header=None
        )
        data[et[2]].x = torch.tensor(df_other.iloc[other_ids, 1:].values).float()
        print(f"num_other_nodes: {data[et[2]].num_nodes} from {df_other.shape[0]}")

        # edges
        row = torch.tensor(user_rev_ids[dfe_selected[0].values])
        col = torch.tensor(other_rev_ids[dfe_selected[1].values])

        # reorder idx
        row_col = row * col.max() + col
        sid = torch.argsort(row_col)

        row = row[sid]
        col = col[sid]

        # put it in graph

        edge_index = torch.stack([row, col])
        data[et].edge_index = edge_index

        xe = torch.tensor(dfe_selected.iloc[:, 2:].values).float()
        xe = xe[sid]
        data[et].edge_attr = xe
        print(f"num_edges: {data[et].num_edges} from {dfe.shape[0]}")

    # store graph
    print("storing graph...")
    torch.save(data, "storage/telecom-small-graph.pt")
    print("DONE")


def synth_random_anomalies():
    # generate nd store data
    import argparse

    parser = argparse.ArgumentParser(description="Hetero_GraphBEAN")
    parser.add_argument(
        "--name", type=str, default="telecom-small-anomaly", help="name"
    )
    parser.add_argument("--n-graph", type=int, default=10, help="n graph")

    args = vars(parser.parse_args())

    graph = torch.load(f"storage/telecom-small-graph.pt")
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
