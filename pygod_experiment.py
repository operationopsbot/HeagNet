# Copyright 2024 Grabtaxi Holdings Pte Ltd (GRAB), All rights reserved.
# Use of this source code is governed by an MIT-style license that can be found in the LICENSE file

import sys

from sklearn.metrics import roc_curve, precision_recall_curve, auc
import numpy as np

import argparse
import os

import torch
from torch_geometric.data import Data
from torch_scatter import scatter

from utils.seed import seed_all

# train a dominant detector
from pygod.detector import DOMINANT, AnomalyDAE, CONAD

# %% args

parser = argparse.ArgumentParser(description="DOMINANT")
parser.add_argument("--name", type=str, default="telecom-small", help="name")
parser.add_argument(
    "--key", type=str, default="graph_anomaly_list", help="key to the data"
)
parser.add_argument("--id", type=int, default=0, help="id to the data")
parser.add_argument("--method", type=str, default="DOMINANT", help="method")

parser.add_argument("--n-epoch", type=int, default=5, help="number of epoch")
parser.add_argument(
    "--num-neighbors", type=int, default=10, help="number of neighbors for node"
)
parser.add_argument("--batch-size", type=int, default=1024, help="batch size")
parser.add_argument("--alpha", type=float, default=0.8, help="balance parameter")
parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
parser.add_argument("--gpu", type=int, default=0, help="gpu number")


args1 = vars(parser.parse_args())

args2 = {
    "seed": 0,
    "hidden_channels": 64,
    "dropout_prob": 0.0,
    "num_layers": 4,
    "verbose": 3,
}

args = {**args1, **args2}

seed_all(args["seed"])

result_dir = "results/"

# %% data
storage = torch.load(f"storage/{args['name']}-anomaly-homogeneous.pt")
dataset = storage[args["key"]][args["id"]]
del storage

# %% model

device = torch.device(f'cuda:{args["gpu"]}' if torch.cuda.is_available() else "cpu")

if args["method"].lower() == "dominant":
    model = DOMINANT(
        hid_dim=args["hidden_channels"],
        num_layers=args["num_layers"],
        dropout=args["dropout_prob"],
        alpha=args["alpha"],
        epoch=args["n_epoch"],
        lr=args["lr"],
        verbose=args["verbose"],
        gpu=args["gpu"],
        batch_size=args["batch_size"],
        num_neigh=args["num_neighbors"],
    )
elif args["method"].lower() == "anomalydae":
    model = AnomalyDAE(
        num_layers=args["num_layers"],
        embed_dim=args["hidden_channels"],
        out_dim=args["hidden_channels"],
        dropout=args["dropout_prob"],
        alpha=args["alpha"],
        epoch=args["n_epoch"],
        lr=args["lr"],
        verbose=args["verbose"],
        gpu=args["gpu"],
        batch_size=args["batch_size"],
        num_neigh=args["num_neighbors"],
    )
elif args["method"].lower() == "conad":
    model = CONAD(
        num_layers=args["num_layers"],
        hid_dim=args["hidden_channels"],
        dropout=args["dropout_prob"],
        alpha=args["alpha"],
        epoch=args["n_epoch"],
        lr=args["lr"],
        verbose=args["verbose"],
        gpu=args["gpu"],
        batch_size=args["batch_size"],
        num_neigh=args["num_neighbors"],
    )
else:
    model = DOMINANT(
        hid_dim=args["hidden_channels"],
        num_layers=args["num_layers"],
        dropout=args["dropout_prob"],
        alpha=args["alpha"],
        epoch=args["n_epoch"],
        lr=args["lr"],
        verbose=args["verbose"],
        gpu=args["gpu"],
        batch_size=args["batch_size"],
        num_neigh=args["num_neighbors"],
    )

print(args)

print(model)

print()


def auc_eval(pred, y):
    rc_curve = roc_curve(y, pred)
    pr_curve = precision_recall_curve(y, pred)
    roc_auc = auc(rc_curve[0], rc_curve[1])
    pr_auc = auc(pr_curve[1], pr_curve[0])

    return roc_auc, pr_auc, rc_curve, pr_curve


# %% run training

print("ready to run")

model.fit(dataset, dataset.y)
score = model.decision_score_

node_result_dict = {}
node_score_dict = {}

print("\nRESULTS >>>")


print("\nnode level:")
for k, v in dataset._node_slices.items():
    node_score = score[v[0] : v[1]].numpy()
    node_label = dataset.y[v[0] : v[1]]

    roc_auc, pr_auc, rc_curve, pr_curve = auc_eval(node_score, node_label)

    node_score_dict[k] = node_score
    node_result_dict[k] = {
        "roc_curve": rc_curve,
        "pr_curve": pr_curve,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }

    print(f"{k}: roc auc: {roc_auc}, pr auc: {pr_auc}")

roc_auc, pr_auc, rc_curve, pr_curve = auc_eval(score.numpy(), dataset.y.numpy())
print(f"global all nodes: roc auc: {roc_auc}, pr auc: {pr_auc}")

# edge level
edge_result_dict = {}
edge_score_dict = {}

score_u = score[dataset.edge_index[0]]
score_v = score[dataset.edge_index[1]]
score_e = (score_u + score_v) / 2

print("\nedge level:")
for k, v in dataset._edge_slices.items():
    edge_score = score_e[v[0] : v[1]].numpy()
    edge_label = dataset.ye[v[0] : v[1]]

    roc_auc, pr_auc, rc_curve, pr_curve = auc_eval(edge_score, edge_label)

    edge_score_dict[k] = edge_score
    edge_result_dict[k] = {
        "roc_curve": rc_curve,
        "pr_curve": pr_curve,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }

    print(f"{k}: roc auc: {roc_auc}, pr auc: {pr_auc}")

roc_auc, pr_auc, rc_curve, pr_curve = auc_eval(score_e.numpy(), dataset.ye.numpy())
print(f"global all edges: roc auc: {roc_auc}, pr auc: {pr_auc}")

# final print
node_avg_roc_auc = np.mean([res["roc_auc"] for res in node_result_dict.values()])
node_avg_pr_auc = np.mean([res["pr_auc"] for res in node_result_dict.values()])
edge_avg_roc_auc = np.mean([res["roc_auc"] for res in edge_result_dict.values()])
edge_avg_pr_auc = np.mean([res["pr_auc"] for res in edge_result_dict.values()])

eval_metrics = {
    "node_result_dict": node_result_dict,
    "edge_result_dict": edge_result_dict,
    "node_avg_roc_auc": node_avg_roc_auc,
    "node_avg_pr_auc": node_avg_pr_auc,
    "edge_avg_roc_auc": edge_avg_roc_auc,
    "edge_avg_pr_auc": edge_avg_pr_auc,
}

print(
    f"--> Metric: "
    + f"node-auc-roc: {eval_metrics['node_avg_roc_auc']:.4f}, edge-auc-roc: {eval_metrics['edge_avg_roc_auc']:.4f}, "
    + f"node-auc-pr {eval_metrics['node_avg_pr_auc']:.4f}, edge-auc-pr {eval_metrics['edge_avg_pr_auc']:.4f} ",
)

output_stored = {
    "args": args,
    "metrics": eval_metrics,
}

torch.save(
    output_stored,
    os.path.join(result_dir, f"{args['method']}-{args['name']}-{args['id']}-output.th"),
)


print("DONE")
