# Copyright 2024 Grabtaxi Holdings Pte Ltd (GRAB), All rights reserved.
# Use of this source code is governed by an MIT-style license that can be found in the LICENSE file

import torch
from torch_geometric.nn.conv import MessagePassing
from torch import ModuleDict, Tensor
import torch.nn as nn
from tqdm import tqdm
import time
import os
import pandas as pd
import numpy as np
import argparse
import copy

from torch_geometric.nn.conv import HeteroConv, GCNConv, SAGEConv, GATConv
from torch_geometric.datasets.fake import FakeHeteroDataset
from torch_sparse import SparseTensor
from torch_geometric.loader import DataLoader, NeighborLoader
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
from models.dataloader import MultiNeighborLoader

from models.net_att import HeagNetAtt
from models.sampler import EdgePredictionSampler, annotate_edge_pred, annotate_target
from models.loss import (
    batch_reconstruction_loss,
    edge_prediction_metric,
    inference_reconstruction_loss,
    reconstruction_loss,
    anomaly_score,
    feature_anomaly_score,
    top_k_features,
    compute_evaluation_metrics,
)
from utils.combine import combine_rev_edges

from utils.seed import seed_all
from utils.standardize import standardize_features

# %% args
parser = argparse.ArgumentParser(description="HeagNet")
parser.add_argument("--id", type=int, default=0, help="id to the data")
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--n-epoch", type=int, default=10, help="number of epoch")

parser.add_argument(
    "--scheduler-milestones",
    nargs="+",
    type=int,
    default=[5, 8],
    help="number of epoch",
)
args1 = vars(parser.parse_args())

args2 = {
    "name": f"telecom-small",
    "n_layers_encoder": 2,
    "n_layers_decoder": 2,
    "n_layers_mlp": 2,
    "sampling-n-hops": 3,
    "gamma": 0.2,
    "x_loss_weight": 1.0,
    "xe_loss_weight": 1.0,
    "structure_loss_weight": 0.2,
    "edge_pred_mult": 3.0,
    "min_num_edge": 2,
    "num_workers": 8,
    "clip_max": 100.0,
    "seed": 0,
    "progress_bar": True,
    "iter_check": 1,
}

args = {**args1, **args2}

seed_all(args["seed"])

# %% dataset

result_dir = "results/"

# load datasets
storage = torch.load("storage/telecom-small-anomaly.pt")
dataset = storage["graph_anomaly_list"][args["id"]]

# transform
# remove isolated nodes
transform = T.RemoveIsolatedNodes()
dataset = transform(dataset)


# add ids to graph
for nt in dataset.node_types:
    dataset[nt]["df_nid"] = torch.arange(dataset[nt].num_nodes)

for et in dataset.edge_types:
    dataset[et]["df_eid"] = torch.arange(dataset[et].num_edges)

# get metadata
metadata = dataset.metadata()
ntypes, etypes = metadata

# standardize
dataset = standardize_features(dataset)

print(dataset)
print("\nAnomalies -->")
for nt in ntypes:
    print(
        f"{nt}: {dataset[nt].y.sum()}/{dataset[nt].y.shape[0]}  ({dataset[nt].y.sum()/dataset[nt].y.shape[0]})"
    )
for et in etypes:
    print(
        f"{et}: {dataset[et].ye.sum()}/{dataset[et].ye.shape[0]}  ({dataset[et].ye.sum()/dataset[et].ye.shape[0]})"
    )
print("\n")


# model
in_channels_node = {key: dataset[key].x.shape[1] for key in ntypes}
in_channels_edge = {key: dataset[key].edge_attr.shape[1] for key in etypes}

hidden_channels_node = {key: 32 for key in ntypes}
hidden_channels_edge = {key: 32 for key in etypes}

latent_channels_node = {key: 64 for key in ntypes}
edge_pred_latent = {key: 64 for key in etypes}

print("model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HeagNetAtt(
    metadata,
    in_channels_node,
    in_channels_edge,
    n_layers_encoder=args["n_layers_encoder"],
    n_layers_decoder=args["n_layers_decoder"],
    n_layers_mlp=args["n_layers_mlp"],
    bidirectional=True,
)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args["lr"])
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=args["scheduler_milestones"], gamma=args["gamma"]
)


# transform graph for sampling
transform = T.ToUndirected()
dataset_sampling = transform(copy.copy(dataset))

# neighbor sampling
num_neighbors_dict = {
    ("user", "buy", "package"): 10,
    ("user", "live", "cell"): 10,
    ("user", "use", "app"): 5,
    ("package", "rev_buy", "user"): 10,
    ("cell", "rev_live", "user"): 10,
    ("app", "rev_use", "user"): 10,
}

# loader & sampler
n_step = args["sampling-n-hops"]

dataloader = MultiNeighborLoader(
    dataset_sampling,
    num_neighbors={
        key: [num_neighbors_dict[key]] * n_step for key in dataset_sampling.edge_types
    },
    input_nodes=["user", "app", "cell", "package"],
    batch_size=[1024, 256, 512, 256],
    # batch_size=[256, 256, 256, 256],
    # batch_size=[32, 32, 32, 32],
    # input_repeat=[1.0, 1.0, 0.2, 1.0],
    input_repeat=[0.8, 0.8, 0.4, 0.8],
    drop_last=True,
)

node_score_agg = {
    "user": "max",
    "package": "max",
    "cell": "max",
    "app": "max",
}

# print(dataset_sampling)


def train(epoch):
    model.train()

    n_batch = len(dataloader)
    if args["progress_bar"]:
        pbar = tqdm(total=n_batch, leave=False)
        pbar.set_description(f"#{epoch:3d}")
    else:
        start = time.time()

    for i, batch in enumerate(dataloader):
        batch = combine_rev_edges(batch)
        batch = batch.to(device)

        x_dict = {
            key: batch[key].x
            for key in batch.metadata()[0]
            if batch[key].x.shape[0] > 0
        }
        spt_dict = {
            key: SparseTensor.from_edge_index(
                batch[key].edge_index,
                sparse_sizes=(batch[key[0]].num_nodes, batch[key[2]].num_nodes),
                edge_attr=batch[key].edge_attr,
            )
            for key in batch.metadata()[1]
            if batch[key].edge_index.shape[1] >= args["min_num_edge"]
        }
        adj_dict = {k: spt_dict[k].set_value(None) for k in spt_dict.keys()}
        xe_dict = {k: spt_dict[k].storage.value() for k in spt_dict.keys()}

        edge_pred_sampler_dict = {}
        edge_pred_samples_dict = {}

        for et, adj in adj_dict.items():
            if adj.sparse_sizes()[0] > 0 and adj.sparse_sizes()[1] > 0:
                edge_pred_sampler = EdgePredictionSampler(
                    adj, mult=args["edge_pred_mult"]
                )
                edge_pred_samples = edge_pred_sampler.sample()

                edge_pred_sampler_dict[et] = edge_pred_sampler
                edge_pred_samples_dict[et] = edge_pred_samples

        # annotate
        # batch = annotate_target(batch)
        # edge_pred_target_mask_dict = annotate_edge_pred(edge_pred_samples_dict, batch)

        # annotate
        for nt in batch.node_types:
            batch[nt].target_node = torch.ones(
                batch[nt].num_nodes, dtype=torch.bool, device=batch[nt].x.device
            )
        for et in batch.edge_types:
            batch[et].target_edge = torch.ones(
                batch[et].num_edges,
                dtype=torch.bool,
                device=batch[et].edge_index.device,
            )
        edge_pred_target_mask_dict = {}
        for et, adj in adj_dict.items():
            edge_pred_target_mask_dict[et] = torch.ones(
                edge_pred_samples_dict[et].nnz(), dtype=torch.bool
            )

        # forward
        optimizer.zero_grad()
        out = model(x_dict, xe_dict, adj_dict, edge_pred_samples_dict)

        # loss
        loss, loss_component = batch_reconstruction_loss(
            batch,
            out,
            edge_pred_samples_dict,
            edge_pred_target_mask_dict,
            x_loss_weight=args["x_loss_weight"],
            xe_loss_weight=args["xe_loss_weight"],
            structure_loss_weight=args["structure_loss_weight"],
        )

        epred_metric = edge_prediction_metric(
            edge_pred_samples_dict, edge_pred_target_mask_dict, out["eprob_dict"]
        )

        # backward
        loss.backward()
        optimizer.step()

        if args["progress_bar"]:
            pbar.update(1)
            pbar.set_postfix(
                {
                    "loss": float(loss),
                    "l xe": float(loss_component["xe"]),
                    "ep acc": epred_metric["acc"],
                    "ep f1": epred_metric["f1"],
                }
            )
        else:
            if i % args["k_check"] == 0 and i != 0:
                elapsed = time.time() - start
                print(
                    f"#{epoch:3d} ({i}/{n_batch}), "
                    + f"Loss: {loss:.4f} => x: {loss_component['x']:.4f}, "
                    + f"xe: {loss_component['xe']:.4f}, "
                    + f"e: {loss_component['e']:.4f} -> "
                    + f"[acc: {epred_metric['acc']:.3f}, f1: {epred_metric['f1']:.3f} -> "
                    + f"prec: {epred_metric['prec']:.3f}, rec: {epred_metric['rec']:.3f}] "
                    + f"| {elapsed:.2f}s",
                    flush=True,
                )
                start = time.time()

    if args["progress_bar"]:
        pbar.close()

    scheduler.step()

    return loss, loss_component, epred_metric


def eval():
    model.eval()

    tz = time.time()

    # inference
    node_batch_size = {key: 2**14 if key in ["cell"] else 2**13 for key in ntypes}
    mlp_batch_size = {key: 2**14 for key in ntypes}
    epred_batch_size = {key: 2**14 for key in etypes}

    with torch.no_grad():
        # print("forward", flush=True)
        out = model.inference(
            dataset,
            node_batch_size=node_batch_size,
            mlp_batch_size=mlp_batch_size,
            epred_batch_size=epred_batch_size,
            device=device,
            edge_pred_mult=args["edge_pred_mult"],
            progress_bar=args["progress_bar"],
        )

        # print("loss", flush=True)
        loss, loss_component = inference_reconstruction_loss(
            dataset,
            out,
            xe_loss_weight=args["xe_loss_weight"],
            structure_loss_weight=args["structure_loss_weight"],
        )

        # print("epred", flush=True)
        # for et, adj in adj_dict.items():
        edge_pred_target_mask_dict = {}
        for et, edge_pred_samples in out["edge_pred_samples_dict"].items():
            edge_pred_target_mask_dict[et] = torch.ones(
                edge_pred_samples.nnz(), dtype=torch.bool
            )

        epred_metric = edge_prediction_metric(
            out["edge_pred_samples_dict"], edge_pred_target_mask_dict, out["eprob_dict"]
        )

        score = anomaly_score(
            dataset,
            out,
            xe_loss_weight=args["xe_loss_weight"],
            structure_loss_weight=args["structure_loss_weight"],
            bidirectional=True,
        )

        eval_metrics = compute_evaluation_metrics(dataset, score, agg=node_score_agg)

    print(
        f"#Eval "
        + f"Loss: {loss:.4f} => x: {loss_component['x']:.4f}, "
        + f"xe: {loss_component['xe']:.4f}, "
        + f"e: {loss_component['e']:.4f} -> "
        + f"[acc: {epred_metric['acc']:.3f}, f1: {epred_metric['f1']:.3f} -> "
        + f"prec: {epred_metric['prec']:.3f}, rec: {epred_metric['rec']:.3f}] ",
        flush=True,
    )

    print(
        f"      --> Metric: "
        + f"node-auc-roc: {eval_metrics['node_avg_roc_auc']:.4f}, edge-auc-roc: {eval_metrics['edge_avg_roc_auc']:.4f}, "
        + f"node-auc-pr {eval_metrics['node_avg_pr_auc']:.4f}, edge-auc-pr {eval_metrics['edge_avg_pr_auc']:.4f} "
        + f"| {time.time()-tz:.4f}s",
    )

    ## each node / edge
    for nt in ntypes:
        print(f"{nt} x_loss: {loss_component['x_loss_dict'][nt]}")
    for et in etypes:
        print(
            f"{et}: xe_loss: {loss_component['xe_loss_dict'][et]}, structure_loss: {loss_component['structure_loss_dict'][et]}"
        )

    print()

    ## each node / edge
    for nt in ntypes:
        print(
            f"{nt} ({node_score_agg[nt]}): roc auc: {eval_metrics['node_result_dict'][nt]['roc_auc']}, pr auc: {eval_metrics['node_result_dict'][nt]['pr_auc']}"
        )
    for et in etypes:
        print(
            f"{et}: roc auc: {eval_metrics['edge_result_dict'][et]['roc_auc']}, pr auc: {eval_metrics['edge_result_dict'][et]['pr_auc']}"
        )

    print()

    model_stored = {
        "args": args,
        "loss": loss,
        "loss_component": loss_component,
        "epred_metric": epred_metric,
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    output_stored = {"args": args, "metrics": eval_metrics}

    torch.save(
        model_stored,
        os.path.join(result_dir, f"graphbeanatt-{args['name']}-{args['id']}-model.th"),
    )
    torch.save(
        output_stored,
        os.path.join(result_dir, f"graphbeanatt-{args['name']}-{args['id']}-output.th"),
    )


eval()
for epoch in range(args["n_epoch"]):
    start = time.time()
    loss, loss_component, epred_metric = train(epoch)
    elapsed = time.time() - start

    print(
        f"#{epoch:3d}, "
        + f"Loss: {loss:.4f} => x: {loss_component['x']:.4f}, "
        + f"xe: {loss_component['xe']:.4f}, "
        + f"e: {loss_component['e']:.4f} -> "
        + f"[acc: {epred_metric['acc']:.3f}, f1: {epred_metric['f1']:.3f} -> "
        + f"prec: {epred_metric['prec']:.3f}, rec: {epred_metric['rec']:.3f}] "
        + f"| {elapsed:.2f}s",
        flush=True,
    )

    if epoch % args["iter_check"] == 0 or epoch == args["n_epoch"] - 1:
        # tb eval
        eval()

# eval()
print(f">> graphbeanatt-{args['name']}-{args['id']} >> DONE >>")
