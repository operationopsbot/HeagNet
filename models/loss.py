# Copyright 2024 Grabtaxi Holdings Pte Ltd (GRAB), All rights reserved.
# Use of this source code is governed by an MIT-style license that can be found in the LICENSE file

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np

from torch_scatter import scatter
from torch_sparse import SparseTensor
from torch_geometric.data import HeteroData
from collections import defaultdict


from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    precision_recall_curve,
    auc,
)

from torch_geometric.typing import (
    PairTensor,
    OptTensor,
    NodeType,
    EdgeType,
    Adj,
    Metadata,
)


def reconstruction_loss(
    graph: HeteroData,
    out: Dict[str, Dict],
    edge_pred_samples_dict: Dict[EdgeType, SparseTensor],
    x_loss_weight: float = 1.0,
    xe_loss_weight: float = 1.0,
    structure_loss_weight: float = 1.0,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    # feature mse
    x_loss_dict = {
        k: torch.mean((graph[k].x - out["x_dict"][k]) ** 2, dim=1)
        for k in graph.node_types
    }

    x_loss = torch.mean(torch.cat(list(x_loss_dict.values())))

    xe_loss_dict = {
        k: torch.mean((graph[k].edge_attr - out["xe_dict"][k]) ** 2, dim=1)
        for k in graph.edge_types
    }
    xe_loss = torch.mean(torch.cat(list(xe_loss_dict.values())))

    feature_loss = x_loss_weight * x_loss + xe_loss_weight * xe_loss

    # structure loss
    structure_loss_dict = {
        k: F.binary_cross_entropy(
            out["eprob_dict"][k], (s.storage.value() > 0).float(), reduction="none"
        )
        for k, s in edge_pred_samples_dict.items()
    }

    structure_loss = torch.mean(torch.cat(list(structure_loss_dict.values())))

    loss = feature_loss + structure_loss_weight * structure_loss

    loss_component = {
        "x": x_loss,
        "xe": xe_loss,
        "e": structure_loss,
        "total": loss,
        "x_loss_dict": x_loss_dict,
        "xe_loss_dict": xe_loss_dict,
        "structure_loss_dict": structure_loss_dict,
    }

    return loss, loss_component


def inference_reconstruction_loss(
    graph: HeteroData,
    out: Dict[str, Dict],
    x_loss_weight: float = 1.0,
    xe_loss_weight: float = 1.0,
    structure_loss_weight: float = 1.0,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    return reconstruction_loss(
        graph,
        out=out,
        edge_pred_samples_dict=out["edge_pred_samples_dict"],
        x_loss_weight=x_loss_weight,
        xe_loss_weight=xe_loss_weight,
        structure_loss_weight=structure_loss_weight,
    )


def batch_reconstruction_loss(
    batch: HeteroData,
    out: Dict[str, Dict],
    edge_pred_samples_dict: Dict[EdgeType, SparseTensor],
    edge_pred_target_mask_dict: Dict[EdgeType, Tensor],
    x_loss_weight: float = 1.0,
    xe_loss_weight: float = 1.0,
    structure_loss_weight: float = 1.0,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    # feature mse
    x_loss_list = []
    for nt in out["x_dict"].keys():
        if torch.sum(batch[nt].target_node) > 0:
            x = batch[nt].x[batch[nt].target_node]
            x_hat = out["x_dict"][nt][batch[nt].target_node]
            x_loss_list.append(torch.mean((x - x_hat) ** 2, dim=1))

    x_loss = torch.mean(torch.cat(x_loss_list))

    xe_loss_list = []
    for et in out["xe_dict"].keys():
        if torch.sum(batch[et].target_edge) > 0:
            xe = batch[et].edge_attr[batch[et].target_edge]
            xe_hat = out["xe_dict"][et][batch[et].target_edge]
            xe_loss_list.append(torch.mean((xe - xe_hat) ** 2, dim=1))

    xe_loss = torch.mean(torch.cat(xe_loss_list))

    feature_loss = x_loss_weight * x_loss + xe_loss_weight * xe_loss

    # structure loss
    structure_loss_list = []
    for k, s in edge_pred_samples_dict.items():
        if torch.sum(batch[k].target_edge) > 0:
            eprob = out["eprob_dict"][k][edge_pred_target_mask_dict[k]]
            target = (s.storage.value() > 0).float()[edge_pred_target_mask_dict[k]]
            loss = F.binary_cross_entropy(eprob, target, reduction="none")
            structure_loss_list.append(loss)
    structure_loss = torch.mean(torch.cat(structure_loss_list))

    loss = feature_loss + structure_loss_weight * structure_loss

    loss_component = {
        "x": x_loss,
        "xe": xe_loss,
        "e": structure_loss,
        "total": loss,
    }

    return loss, loss_component


def edge_prediction_metric(
    edge_pred_samples_dict: Dict[EdgeType, SparseTensor],
    edge_pred_target_mask_dict: Dict[EdgeType, Tensor],
    edge_prob_dict: Dict[EdgeType, Tensor],
) -> Dict:
    acc_dict = {}
    prec_dict = {}
    rec_dict = {}
    f1_dict = {}
    for k in edge_pred_samples_dict.keys():
        if torch.sum(edge_pred_target_mask_dict[k]) > 0:
            edge_pred = (
                (edge_prob_dict[k][edge_pred_target_mask_dict[k]] >= 0.5)
                .int()
                .cpu()
                .numpy()
            )
            edge_gt = (
                (
                    edge_pred_samples_dict[k].storage.value()[
                        edge_pred_target_mask_dict[k]
                    ]
                    > 0
                )
                .int()
                .cpu()
                .numpy()
            )

            acc = accuracy_score(edge_gt, edge_pred)
            prec = precision_score(edge_gt, edge_pred, zero_division=0)
            rec = recall_score(edge_gt, edge_pred, zero_division=0)
            f1 = f1_score(edge_gt, edge_pred, zero_division=0)

            acc_dict[k] = acc
            prec_dict[k] = prec
            rec_dict[k] = rec
            f1_dict[k] = f1

    acc = np.mean(list(acc_dict.values()))
    prec = np.mean(list(prec_dict.values()))
    rec = np.mean(list(rec_dict.values()))
    f1 = np.mean(list(f1_dict.values()))

    result = {
        "acc": acc,
        "prec": prec,
        "rec": rec,
        "f1": f1,
        "acc_dict": acc_dict,
        "prec_dict": prec_dict,
        "rec_dict": rec_dict,
        "f1_dict": f1_dict,
    }
    return result


def anomaly_score(
    graph: HeteroData,
    out: Dict[str, Dict],
    x_loss_weight: float = 1.0,
    xe_loss_weight: float = 1.0,
    structure_loss_weight: float = 1.0,
    bidirectional: bool = False,
) -> Dict[str, Tensor]:
    # get data
    x_dict = {k: graph[k].x for k in graph.node_types}
    xe_dict = {k: graph[k].edge_attr for k in graph.edge_types}
    edge_index_dict = {k: graph[k].edge_index for k in graph.edge_types}
    edge_pred_samples_dict = out["edge_pred_samples_dict"]

    # feature error
    x_error = {
        k: x_loss_weight * torch.sqrt(torch.mean((x - out["x_dict"][k]) ** 2, dim=1))
        for k, x in x_dict.items()
    }

    xe_error = {
        k: torch.sqrt(torch.mean((xe - out["xe_dict"][k]) ** 2, dim=1))
        for k, xe in xe_dict.items()
    }

    # structure loss
    edge_ce = {
        k: -torch.log(out["eprob_dict"][k][s.storage.value() > 0] + 1e-12)
        for k, s in edge_pred_samples_dict.items()
    }

    # edge score
    edge_score = {
        k: xe_loss_weight * xe_error[k] + structure_loss_weight * edge_ce[k]
        for k in xe_dict.keys()
    }

    error_max_list_dict = defaultdict(list)
    error_mean_list_dict = defaultdict(list)
    for et, edge_index in edge_index_dict.items():
        src, rel, dst = et
        error_max_list_dict[dst].append(
            scatter(
                edge_score[et],
                edge_index[1],
                dim=0,
                dim_size=graph[dst].num_nodes,
                reduce="max",
            )
        )
        error_mean_list_dict[dst].append(
            scatter(
                edge_score[et],
                edge_index[1],
                dim=0,
                dim_size=graph[dst].num_nodes,
                reduce="mean",
            )
        )

        if bidirectional:
            error_max_list_dict[src].append(
                scatter(
                    edge_score[et],
                    edge_index[0],
                    dim=0,
                    dim_size=graph[src].num_nodes,
                    reduce="max",
                )
            )
            error_mean_list_dict[src].append(
                scatter(
                    edge_score[et],
                    edge_index[0],
                    dim=0,
                    dim_size=graph[src].num_nodes,
                    reduce="mean",
                )
            )

    node_score_edge_max = {}
    node_score_edge_mean = {}
    for k in x_dict.keys():
        node_score_edge_max[k] = x_error[k].clone()
        if len(error_max_list_dict) > 0:
            node_score_edge_max[k] += torch.max(
                torch.stack(error_max_list_dict[k], dim=1), dim=1
            )[0]

        node_score_edge_mean[k] = x_error[k].clone()
        if len(error_mean_list_dict) > 0:
            node_score_edge_mean[k] += torch.mean(
                torch.stack(error_mean_list_dict[k], dim=1), dim=1
            )

    score = {
        "x_error": x_error,
        "xe_error": xe_error,
        "edge_ce": edge_ce,
        "edge_score": edge_score,
        "node_score_edge_max": node_score_edge_max,
        "node_score_edge_mean": node_score_edge_mean,
    }

    return score


def feature_anomaly_score(
    graph: HeteroData,
    out: Dict[str, Dict],
    x_loss_weight: float = 1.0,
    xe_loss_weight: float = 1.0,
    structure_loss_weight: float = 1.0,
) -> Dict[str, Tensor]:
    # get data
    x_dict = {k: graph[k].x for k in graph.node_types}
    xe_dict = {k: graph[k].edge_attr for k in graph.edge_types}
    edge_index_dict = {k: graph[k].edge_index for k in graph.edge_types}
    edge_pred_samples_dict = out["edge_pred_samples_dict"]

    # feature error
    # sqrt of **2 = abs
    x_error = {
        k: x_loss_weight * torch.abs(x - out["x_dict"][k]) for k, x in x_dict.items()
    }
    xe_error = {
        k: xe_loss_weight * torch.abs(xe - out["xe_dict"][k])
        for k, xe in xe_dict.items()
    }

    # structure loss
    edge_ce = {
        k: structure_loss_weight
        * (-torch.log(out["eprob_dict"][k][s.storage.value() > 0] + 1e-12))
        for k, s in edge_pred_samples_dict.items()
    }

    score = {
        "x_error": x_error,
        "xe_error": xe_error,
        "edge_ce": edge_ce,
    }

    return score


def top_k_features(
    graph: HeteroData,
    out: Dict[str, Dict],
    node_feat_meta: Dict[str, np.chararray],
    edge_feat_meta: Dict[str, np.chararray],
    x_loss_weight: float = 1.0,
    xe_loss_weight: float = 1.0,
    k: int = 10,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    # get data
    x_dict = {key: graph[key].x for key in graph.node_types}
    xe_dict = {key: graph[key].edge_attr for key in graph.edge_types}
    edge_index_dict = {key: graph[key].edge_index for key in graph.edge_types}

    # top k edge
    xe_error = {}
    edge_top_k = {}
    for et in graph.edge_types:
        # error
        err = xe_loss_weight * torch.abs(xe_dict[et] - out["xe_dict"][et])
        idx = torch.argsort(err, dim=1, descending=True)
        idk = idx[:, :k]
        top_feat = edge_feat_meta[et][idk.flatten()].reshape(idk.shape)

        edge_top_k[et] = np.apply_along_axis(", ".join, 1, top_feat)
        xe_error[et] = err

    # collect edge error
    error_max_list_dict = defaultdict(list)
    error_mean_list_dict = defaultdict(list)
    feat_max_list_dict = defaultdict(list)
    feat_mean_list_dict = defaultdict(list)
    for et, edge_index in edge_index_dict.items():
        src, rel, dst = et

        err_max = scatter(
            xe_error[et],
            edge_index[1],
            dim=0,
            dim_size=graph[dst].num_nodes,
            reduce="max",
        )
        idx = torch.argsort(err_max, dim=1, descending=True)
        idk = idx[:, :k]
        top_feat_max = edge_feat_meta[et][idk.flatten()].reshape(idk.shape)
        top_err_max = torch.take_along_dim(err_max, idk, dim=1)

        err_mean = scatter(
            xe_error[et],
            edge_index[1],
            dim=0,
            dim_size=graph[dst].num_nodes,
            reduce="mean",
        )
        idx = torch.argsort(err_mean, dim=1, descending=True)
        idk = idx[:, :k]
        top_feat_mean = edge_feat_meta[et][idk.flatten()].reshape(idk.shape)
        top_err_mean = torch.take_along_dim(err_mean, idk, dim=1)

        error_max_list_dict[dst].append(top_err_max)
        error_mean_list_dict[dst].append(top_err_mean)

        feat_max_list_dict[dst].append(rel + "." + top_feat_max)
        feat_mean_list_dict[dst].append(rel + "." + top_feat_mean)

    # node top k
    node_top_k_max = {}
    node_top_k_mean = {}
    for nt in graph.node_types:
        x_err = x_loss_weight * torch.abs(x_dict[nt] - out["x_dict"][nt])

        node_err_edge_max = torch.cat(
            [x_err, torch.cat(error_max_list_dict[nt], dim=1)], dim=1
        )
        node_err_edge_mean = torch.cat(
            [x_err, torch.cat(error_mean_list_dict[nt], dim=1)], dim=1
        )

        feat_meta_max = np.concatenate(
            [node_feat_meta[nt], np.concatenate(feat_max_list_dict[nt])]
        )

        feat_meta_mean = np.concatenate(
            [node_feat_meta[nt], np.concatenate(feat_mean_list_dict[nt])]
        )

        idx = torch.argsort(node_err_edge_max, dim=1, descending=True)
        idk = idx[:, :k]
        top_feat_max = feat_meta_max[idk.flatten()].reshape(idk.shape)
        node_top_k_max[nt] = np.apply_along_axis(", ".join, 1, top_feat_max)

        idx = torch.argsort(node_err_edge_mean, dim=1, descending=True)
        idk = idx[:, :k]
        top_feat_mean = feat_meta_mean[idk.flatten()].reshape(idk.shape)
        node_top_k_mean[nt] = np.apply_along_axis(", ".join, 1, top_feat_mean)

    topk = {
        "edge_top_k": edge_top_k,
        "node_top_k_mean": node_top_k_mean,
        "node_top_k_max": node_top_k_max,
    }

    return topk


def compute_evaluation_metrics(
    graph: HeteroData,
    score: Dict[str, Tensor],
    agg: Union[str, Dict[str, str]] = "max",
    include_rev_edge: bool = False,
):
    node_result_dict = {}
    for nt in graph.node_types:
        ag = agg[nt] if isinstance(agg, Dict) else agg

        if ag == "none":
            sc = score[f"x_error"][nt].cpu().numpy()
        else:
            sc = score[f"node_score_edge_{ag}"][nt].cpu().numpy()

        node_roc_curve = roc_curve(graph[nt].y.cpu().numpy(), sc)
        node_pr_curve = precision_recall_curve(graph[nt].y.cpu().numpy(), sc)
        roc_auc = auc(node_roc_curve[0], node_roc_curve[1])
        pr_auc = auc(node_pr_curve[1], node_pr_curve[0])

        node_result_dict[nt] = {
            "roc_curve": node_roc_curve,
            "pr_curve": node_pr_curve,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
        }

    edge_result_dict = {}
    for et in graph.edge_types:
        if include_rev_edge or (not et[1].startswith("rev_")):
            edge_roc_curve = roc_curve(
                graph[et].ye.cpu().numpy(), score[f"xe_error"][et].cpu().numpy()
            )
            edge_pr_curve = precision_recall_curve(
                graph[et].ye.cpu().numpy(), score[f"xe_error"][et].cpu().numpy()
            )
            roc_auc = auc(edge_roc_curve[0], edge_roc_curve[1])
            pr_auc = auc(edge_pr_curve[1], edge_pr_curve[0])

            edge_result_dict[et] = {
                "roc_curve": edge_roc_curve,
                "pr_curve": edge_pr_curve,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
            }

    node_avg_roc_auc = np.mean([res["roc_auc"] for res in node_result_dict.values()])
    node_avg_pr_auc = np.mean([res["pr_auc"] for res in node_result_dict.values()])
    edge_avg_roc_auc = np.mean([res["roc_auc"] for res in edge_result_dict.values()])
    edge_avg_pr_auc = np.mean([res["pr_auc"] for res in edge_result_dict.values()])

    metrics = {
        "node_result_dict": node_result_dict,
        "edge_result_dict": edge_result_dict,
        "node_avg_roc_auc": node_avg_roc_auc,
        "node_avg_pr_auc": node_avg_pr_auc,
        "edge_avg_roc_auc": edge_avg_roc_auc,
        "edge_avg_pr_auc": edge_avg_pr_auc,
    }

    return metrics
