# Copyright 2024 Grabtaxi Holdings Pte Ltd (GRAB), All rights reserved.
# Use of this source code is governed by an MIT-style license that can be found in the LICENSE file

import sys

from sklearn.metrics import roc_curve, precision_recall_curve, auc

from torch_scatter import scatter
from torch_sparse import SparseTensor
from torch_geometric.data import HeteroData
from collections import defaultdict

import torch_geometric.transforms as T

import argparse
import os

import torch
from utils.seed import seed_all
import numpy as np

from sklearn.ensemble import IsolationForest
from utils.standardize import standardize_features


# %% args

parser = argparse.ArgumentParser(description="IsolationForest")
parser.add_argument("--name", type=str, default="telecom-small", help="name")
parser.add_argument(
    "--key", type=str, default="graph_anomaly_list", help="key to the data"
)
parser.add_argument("--id", type=int, default=0, help="id to the data")

args1 = vars(parser.parse_args())

args2 = {
    "seed": 1,
}

args = {**args1, **args2}

seed_all(args["seed"])

result_dir = "results/"


# %% data
storage = torch.load(f"storage/{args['name']}-anomaly.pt")
dataset = storage[args["key"]][args["id"]]
del storage


# transform graph
# transform = T.ToUndirected()
# transform2 = T.RemoveIsolatedNodes()
# dataset = transform2(transform(dataset))

print(dataset)

# get metadata
metadata = dataset.metadata()
ntypes, etypes = metadata

dataset = standardize_features(dataset)


# %% model

print("\n>> LABEL INFO")
for nt in ntypes:
    print(f"{nt}: {dataset[nt].y.sum()}, {dataset[nt].y.sum()/dataset[nt].y.shape[0]}")

for et in etypes:
    print(
        f"{et}: {dataset[et].ye.sum()}, {dataset[et].ye.sum()/dataset[et].ye.shape[0]}"
    )


def train_eval(x, y):
    clf = IsolationForest()
    clf.fit(x)
    score = -clf.score_samples(x)

    rc_curve = roc_curve(y, score)
    pr_curve = precision_recall_curve(y, score)
    roc_auc = auc(rc_curve[0], rc_curve[1])
    pr_auc = auc(pr_curve[1], pr_curve[0])

    return roc_auc, pr_auc, rc_curve, pr_curve


# %% isolation forest

print("\n>> RESULTS")

## node
node_result_dict = {}
for nt in ntypes:
    x = dataset[nt].x
    y = dataset[nt].y
    roc_auc, pr_auc, rc_curve, pr_curve = train_eval(x.numpy(), y.numpy())
    node_result_dict[nt] = {
        "roc_curve": rc_curve,
        "pr_curve": pr_curve,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }
    print(f"{nt}: roc auc: {roc_auc}, pr auc: {pr_auc}")

## edge
edge_result_dict = {}
for et in etypes:
    x = dataset[et].edge_attr
    y = dataset[et].ye
    roc_auc, pr_auc, rc_curve, pr_curve = train_eval(x.numpy(), y.numpy())
    edge_result_dict[et] = {
        "roc_curve": rc_curve,
        "pr_curve": pr_curve,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }
    print(f"{et}: roc auc: {roc_auc}, pr auc: {pr_auc}")

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
print(args)

print()
print(
    f"--> Metric: "
    + f"node-auc-roc: {eval_metrics['node_avg_roc_auc']:.4f}, edge-auc-roc: {eval_metrics['edge_avg_roc_auc']:.4f}, "
    + f"node-auc-pr {eval_metrics['node_avg_pr_auc']:.4f}, edge-auc-pr {eval_metrics['edge_avg_pr_auc']:.4f} ",
)

output_stored = {
    "args": args,
    "metrics": eval_metrics,
}

print("Saving current results...")
torch.save(
    output_stored,
    os.path.join(result_dir, f"isoforest-{args['name']}-{args['id']}-output.th"),
)
