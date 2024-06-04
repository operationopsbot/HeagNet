# Copyright 2024 Grabtaxi Holdings Pte Ltd (GRAB), All rights reserved.
# Use of this source code is governed by an MIT-style license that can be found in the LICENSE file

import torch
import torch_geometric

import numpy as np
from scipy.stats import truncnorm
from torch_sparse import SparseTensor

from torch_geometric.data import HeteroData
from torch_geometric.typing import Adj

from typing import Optional, Tuple, Union, Dict
from torch import Tensor

# %% features outliers


# features outside confidence interval
def outside_cofidence_interval(
    x: Tensor, prop_sample=0.1, prop_feat=0.3, std_cutoff=3.0, mu=None, sigm=None
):
    n, m = x.shape
    ns = int(np.ceil(prop_sample * n))
    ms = int(np.ceil(prop_feat * m))

    # random outlier from truncated normal
    left_side = truncnorm.rvs(-np.inf, -std_cutoff, size=ns * ms)
    right_side = truncnorm.rvs(std_cutoff, np.inf, size=ns * ms)
    lr_flag = np.random.randint(2, size=ns * ms)
    random_outliers = lr_flag * left_side + (1 - lr_flag) * right_side

    # determine which sample & features that are randomized
    feat_idx = np.random.rand(ns, m).argsort(axis=1)[:, :ms]
    sample_idx = np.random.choice(n, ns, replace=False)
    row_idx = np.tile(sample_idx[:, None], (1, ms)).flatten()
    col_idx = feat_idx.flatten()

    # calculate mean and variance
    xr = x.cpu().numpy()
    if mu is None:
        mu = xr.mean(axis=0)
    if sigm is None:
        sigm = xr.std(axis=0)

    # replace the value with outliers
    random_outliers = random_outliers * sigm[col_idx] + mu[col_idx]
    xr[(row_idx, col_idx)] = random_outliers

    # anomaly
    anomaly_label = torch.zeros(n).long()
    anomaly_label[sample_idx] = 1

    return Tensor(xr), anomaly_label, row_idx, col_idx


# add scaled gaussian noise
def scaled_gaussian_noise(
    x: Tensor, scale=3.0, min_dist_rel=3.0, filter=True, mu=None, sigm=None
):
    # calculate mean and variance
    if mu is None:
        mu = x.mean(dim=0)
    if sigm is None:
        sigm = x.std(dim=0)

    # noise
    noise = torch.randn(x.shape) * sigm * scale
    outlier = x + noise
    closest_dist = torch.cdist(outlier, x, p=1).min(dim=1)[0]
    if filter:
        anomaly_label = (closest_dist / x.shape[1] > min_dist_rel).long()
        # replace the value with outliers
        xr = anomaly_label[:, None] * outlier + (1 - anomaly_label[:, None]) * x
    else:
        anomaly_label = torch.ones(x.shape[0]).long()
        xr = outlier

    return xr, anomaly_label


# %% structure outliers
def dense_block(
    adj: Adj,
    xe: Tensor,
    ye=None,
    num_nodes: Union[int, Tuple[int, int]] = 5,
    num_group: int = 2,
    connected_prop=1.0,
    feature_anomaly=False,
    feature_anomaly_type="outside_ci",
    **kwargs,
):
    if isinstance(adj, Tensor):  # edge_index
        n = adj[0].max().item()
        m = adj[1].max().item()
        ids = adj
    else:  # SparseTensor
        n, m = adj.sparse_sizes()

        row = adj.storage.row()
        col = adj.storage.col()
        ids = torch.stack([row, col])

    ne = xe.shape[0]
    if isinstance(num_nodes, int):
        num_nodes = (num_nodes, num_nodes)

    outlier_row = torch.zeros(0).long()
    outlier_col = torch.zeros(0).long()

    for i in range(num_group):
        rid = np.random.choice(n, num_nodes[0], replace=False)
        cid = np.random.choice(m, num_nodes[1], replace=False)

        # all nodes are connected
        rows_id = torch.from_numpy(np.tile(rid[:, None], (1, num_nodes[1])).flatten())
        cols_id = torch.from_numpy(np.tile(cid, num_nodes[0]))

        # partially dense connection
        if connected_prop < 1.0:
            n_connected = rows_id.shape[0]
            n_taken = int(np.ceil(connected_prop * n_connected))
            taken_id = np.random.choice(n_connected, n_taken, replace=False)

            rows_id = rows_id[taken_id]
            cols_id = cols_id[taken_id]

        # add to the graph
        outlier_row = torch.cat([outlier_row, rows_id])
        outlier_col = torch.cat([outlier_col, cols_id])

    # only unique ids
    outlier_ids = torch.stack([outlier_row, outlier_col]).unique(dim=1)

    # find additional ids that is not in the current adj
    ids_all, inv, count = torch.cat([ids, outlier_ids], dim=1).unique(
        dim=1, return_counts=True, return_inverse=True
    )
    ids_duplicate = ids_all[:, count > 1]
    ids_2, count_2 = torch.cat([outlier_ids, ids_duplicate], dim=1).unique(
        dim=1, return_counts=True
    )
    ids_additional = ids_2[:, count_2 == 1]

    # anomalous label for the original
    label_orig = (count[inv][:ne] > 1).long()

    ## features
    n_add = ids_additional.shape[1]
    # random features for the new edges
    add_ids = np.random.choice(ne, n_add, replace=False)
    xe_add = xe[add_ids, :]

    # inject feature anomaly
    xe2 = xe.clone()
    if feature_anomaly:
        # args
        kw2 = {}

        mu = xe.mean(dim=0).numpy()
        sigm = xe.std(dim=0).numpy()
        kw2["mu"] = mu
        kw2["sigm"] = sigm

        if feature_anomaly_type == "outside_ci":
            kw2["prop_sample"] = 1.0
            if "prop_feat_edge" in kwargs:
                kw2["prop_feat"] = kwargs["prop_feat_edge"]
            if "std_cutoff_edge" in kwargs:
                kw2["std_cutoff"] = kwargs["std_cutoff_edge"]
            xe_add = outside_cofidence_interval(xe_add, **kw2)[0]
            if label_orig.sum() > 0:
                xe2[label_orig == 1, :] = outside_cofidence_interval(
                    xe[label_orig == 1, :], **kw2
                )[0]
            else:
                xe2 = xe
        elif feature_anomaly_type == "scaled_gaussian":
            kw2["filter"] = False
            if "scale_edge" in kwargs:
                kw2["scale"] = kwargs["scale_edge"]
            xe_add = scaled_gaussian_noise(xe_add, **kw2)[0]
            if label_orig.sum() > 0:
                xe2[label_orig == 1, :] = scaled_gaussian_noise(
                    xe[label_orig == 1, :], **kw2
                )[0]
            else:
                xe2 = xe

    # combine with the previous label if given
    ye2 = label_orig if ye is None else torch.logical_or(ye, label_orig).long()

    # attach xe and label to value
    ids_cmb = torch.cat([ids, ids_additional], dim=1)
    xe_cmb = torch.cat([xe2, xe_add], dim=0)
    ye_cmb = torch.cat([ye2, torch.ones(n_add).long()])
    label_cmb = torch.cat([label_orig, torch.ones(n_add).long()])
    value_cmb = torch.cat([xe_cmb, ye_cmb[:, None], label_cmb[:, None]], dim=1)

    # get result
    adj_new = SparseTensor(row=ids_cmb[0], col=ids_cmb[1], value=value_cmb).coalesce()
    value_new = adj_new.storage.value()
    xe_new = value_new[:, :-2]
    ye_new = value_new[:, -2].long()
    label = value_new[:, -1].long()
    adj_new.storage._value = None

    if isinstance(adj, Tensor):  # edge_index
        adj_new = torch.stack([adj_new.storage.row(), adj_new.storage.col()])

    return adj_new, xe_new, ye_new, label


# %% graph, insert anomaly
def new_graph(
    data: HeteroData,
    x_dict: Optional[Dict[str, Tensor]] = None,
    xe_dict: Optional[Dict[str, Tensor]] = None,
    adj_dict: Optional[Dict[str, Tensor]] = None,
    y_dict: Optional[Dict[str, Tensor]] = None,
    ye_dict: Optional[Dict[str, Tensor]] = None,
):
    graph = HeteroData()

    for nt in data.node_types:
        if x_dict is not None:
            graph[nt].x = x_dict[nt]
        else:
            graph[nt].x = data[nt].x

        if y_dict is not None:
            graph[nt].y = y_dict[nt]

    for et in data.edge_types:
        if adj_dict is not None:
            graph[et].edge_index = adj_dict[et]
        else:
            graph[et].edge_index = data[et].edge_index

        if xe_dict is not None:
            graph[et].edge_attr = xe_dict[et]
        else:
            graph[et].edge_attr = data[et].edge_attr

        if ye_dict is not None:
            graph[et].ye = ye_dict[et]

    return graph


def inject_feature_anomaly(
    data: HeteroData,
    node_anomaly=True,
    edge_anomaly=True,
    feature_anomaly_type="outside_ci",
    **kwargs,
):
    if node_anomaly:
        x_dict = {}
        y_dict = {}
        for nt in data.node_types:
            # args
            kw2 = {}

            if feature_anomaly_type == "outside_ci":
                if "prop_feat_node" in kwargs:
                    kw2["prop_feat"] = kwargs["prop_feat_node"]
                if "std_cutoff_node" in kwargs:
                    kw2["std_cutoff"] = kwargs["std_cutoff_node"]
                x, y, _, _ = outside_cofidence_interval(data[nt].x, **kw2)
            elif feature_anomaly_type == "scaled_gaussian":
                if "scale_node" in kwargs:
                    kw2["scale"] = kwargs["scale_node"]
                x, y = scaled_gaussian_noise(data[nt].x, **kw2)

            if hasattr(data[nt], "y"):
                y = torch.logical_or(data[nt].y, y).long()

            x_dict[nt] = x
            y_dict[nt] = y
    else:
        x_dict = None
        y_dict = None

    if edge_anomaly:
        xe_dict = {}
        ye_dict = {}
        for et in data.edge_types:
            # args
            kw2 = {}

            if feature_anomaly_type == "outside_ci":
                if "prop_feat_edge" in kwargs:
                    kw2["prop_feat"] = kwargs["prop_feat_edge"]
                if "std_cutoff_edge" in kwargs:
                    kw2["std_cutoff"] = kwargs["std_cutoff_edge"]
                xe, ye, _, _ = outside_cofidence_interval(data[et].edge_attr, **kw2)
            elif feature_anomaly_type == "scaled_gaussian":
                if "scale_edge" in kwargs:
                    kw2["scale"] = kwargs["scale_edge"]
                xe, ye = scaled_gaussian_noise(data[et].edge_attr, **kw2)

            if hasattr(data[et], "ye"):
                ye = torch.logical_or(data[et].ye, ye).long()

            xe_dict[nt] = xe
            ye_dict[nt] = ye

    data_new = new_graph(
        data,
        x_dict=x_dict,
        y_dict=y_dict,
        xe_dict=xe_dict,
        ye_dict=ye_dict,
    )

    return data_new


def inject_dense_block_anomaly(data: HeteroData, num_nodes_dict=None, **kwargs):
    kwargs["feature_anomaly"] = False

    adj_dict = {}
    xe_dict = {}
    ye_dict = {}

    y_dict = {}
    for nt in data.node_types:
        if hasattr(data[nt], "y"):
            y_dict[nt] = data[nt].y

    for et in data.edge_types:
        src, rel, dst = et

        # dense block injection
        ye = data[et].ye if hasattr(data[et], "ye") else None
        adj_new, xe_new, ye_new, label = dense_block(
            data[et].edge_index,
            data[et].edge_attr,
            ye=ye,
            num_nodes=(num_nodes_dict[et][0], num_nodes_dict[et][1]),
            **kwargs,
        )

        adj_dict[et] = adj_new
        xe_dict[et] = xe_new
        ye_dict[et] = ye_new

        # propagate anomaly label to the nodes
        yu = torch.zeros(data[src].x.shape[0]).long()
        yu[adj_new[0][label == 1].unique()] = 1
        yu = torch.logical_or(y_dict[src], yu).long() if src in y_dict else yu

        yv = torch.zeros(data[dst].x.shape[0]).long()
        yv[adj_new[1][label == 1].unique()] = 1
        yv = torch.logical_or(y_dict[dst], yv).long() if dst in y_dict else yv

        y_dict[src] = yu
        y_dict[dst] = yv

    data_new = new_graph(
        data,
        adj_dict=adj_dict,
        y_dict=y_dict,
        xe_dict=xe_dict,
        ye_dict=ye_dict,
    )

    return data_new


def inject_dense_block_and_feature_anomaly(
    data: HeteroData,
    node_feature_anomaly=False,
    edge_feature_anomaly=True,
    num_nodes_dict=None,
    **kwargs,
):
    kwargs["feature_anomaly"] = edge_feature_anomaly
    if "feature_anomaly_type" not in kwargs:
        kwargs["feature_anomaly_type"] = "outside_ci"

    adj_dict = {}
    xe_dict = {}
    ye_dict = {}

    y_dict = {}

    for nt in data.node_types:
        if hasattr(data[nt], "y"):
            y_dict[nt] = data[nt].y

    for et in data.edge_types:
        src, rel, dst = et

        # dense block injection
        ye = data[et].ye if hasattr(data[et], "ye") else None
        adj_new, xe_new, ye_new, label = dense_block(
            data[et].edge_index,
            data[et].edge_attr,
            ye=ye,
            num_nodes=(num_nodes_dict[et][0], num_nodes_dict[et][1]),
            **kwargs,
        )

        adj_dict[et] = adj_new
        xe_dict[et] = xe_new
        ye_dict[et] = ye_new

        # propagate anomaly label to the nodes
        yu = torch.zeros(data[src].x.shape[0]).long()
        yu[adj_new[0][label == 1].unique()] = 1
        yu = torch.logical_or(y_dict[src], yu).long() if src in y_dict else yu

        yv = torch.zeros(data[dst].x.shape[0]).long()
        yv[adj_new[1][label == 1].unique()] = 1
        yv = torch.logical_or(y_dict[dst], yv).long() if dst in y_dict else yv

        y_dict[src] = yu
        y_dict[dst] = yv

    # also node feature anomaly
    if node_feature_anomaly:
        x_dict = {}
        for nt in data.node_types:
            # args
            kw2 = {}

            # xu
            x = data[nt].x
            y = y_dict[nt]
            mu = x.mean(dim=0).numpy()
            sigm = x.std(dim=0).numpy()
            kw2["mu"] = mu
            kw2["sigm"] = sigm

            if kwargs["feature_anomaly_type"] == "outside_ci":
                kw2["prop_sample"] = 1.0
                if "prop_feat_node" in kwargs:
                    kw2["prop_feat"] = kwargs["prop_feat_node"]
                if "std_cutoff_node" in kwargs:
                    kw2["std_cutoff"] = kwargs["std_cutoff_node"]
                x_new = x.clone()
                x_new[y == 1, :] = outside_cofidence_interval(x[y == 1, :], **kw2)[0]
            elif kwargs["feature_anomaly_type"] == "scaled_gaussian":
                kw2["filter"] = False
                if "scale_node" in kwargs:
                    kw2["scale"] = kwargs["scale_node"]
                if "min_dist_rel" in kwargs:
                    kw2["min_dist_rel"] = kwargs["min_dist_rel"]
                x_new = x.clone()
                x_new[y == 1, :] = scaled_gaussian_noise(x[y == 1, :], **kw2)[0]

            x_dict[nt] = x_new
    else:
        x_dict = None

    data_new = new_graph(
        data,
        adj_dict=adj_dict,
        x_dict=x_dict,
        y_dict=y_dict,
        xe_dict=xe_dict,
        ye_dict=ye_dict,
    )

    return data_new


# %% random anomaly


def choose(r, choices, thresholds):
    i = 0
    cm = thresholds[i]
    while i < len(choices):
        if r <= cm + 1e-9:
            selected = i
            break
        else:
            i += 1
            if i < len(choices):
                cm += thresholds[i]
            else:
                selected = len(choices) - 1
                break

    return choices[selected]


def inject_random_block_anomaly(
    data: HeteroData,
    num_group=40,
    num_nodes_range_dict=None,
    **kwargs,
):
    block_anomalies = ["full_dense_block", "partial_full_dense_block"]  # , 'none']
    feature_anomalies = ["outside_ci", "scaled_gaussian", "none"]
    node_edge_feat_anomalies = ["node_only", "edge_only", "node_edge"]

    # block_anomalies_weight = [0.2, 0.8]  # , 0.1]
    # feature_anomalies_weight = [0.5, 0.4, 0.1]
    # node_edge_feat_anomalies_weight = [0.1, 0.3, 0.6]

    block_anomalies_weight = [0.3, 0.7]  # , 0.1]
    feature_anomalies_weight = [0.4, 0.4, 0.2]
    # node_edge_feat_anomalies_weight = [0.1, 0.4, 0.5]
    # node_edge_feat_anomalies_weight = [0.3, 0.2, 0.5]
    node_edge_feat_anomalies_weight = [0.1, 0.3, 0.6]

    data_new = new_graph(data)

    # random anomaly
    for itg in range(num_group):
        print(f"it {itg}: ", end="")

        rnd = torch.rand(3)
        block_an = choose(rnd[0], block_anomalies, block_anomalies_weight)
        feature_an = choose(rnd[1], feature_anomalies, feature_anomalies_weight)
        node_edge_an = choose(
            rnd[2], node_edge_feat_anomalies, node_edge_feat_anomalies_weight
        )

        num_nodes_dict = {}
        for et in data.edge_types:
            num_nodes_list = []
            for num_nodes_range in num_nodes_range_dict[et]:
                lr, rr, mr = (
                    num_nodes_range[0],
                    num_nodes_range[1],
                    num_nodes_range[0] + num_nodes_range[1] / 2,
                )
                nn = int(
                    np.minimum(
                        np.maximum(lr, (torch.randn(1).item() * np.sqrt(mr)) + mr),
                        rr + 1,
                    )
                )
                num_nodes_list.append(nn)
            num_nodes_dict[et] = tuple(num_nodes_list)

        ## setup kwargs
        connected_prop = 1.0
        if block_an == "partial_full_dense_block":
            connected_prop = np.minimum(
                np.maximum(0.2, (torch.randn(1).item() / 4) + 0.5), 1.0
            )

            # connected_prop = np.minimum(
            #     np.maximum(0.5, (torch.randn(1).item() / 4) + 0.5), 1.0
            # )

        prop_feat_edge = np.minimum(
            np.maximum(0.1, (torch.randn(1).item() / 8) + 0.3), 0.9
        )
        std_cutoff_edge = np.maximum(2.0, torch.randn(1).item() + 3.0)
        scale_edge = np.maximum(2.0, torch.randn(1).item() + 3.0)

        prop_feat_node = np.minimum(
            np.maximum(0.1, (torch.randn(1).item() / 8) + 0.2), 0.5
        )
        std_cutoff_node = np.maximum(1.5, torch.randn(1).item() + 2.0)
        scale_node = np.maximum(1.5, torch.randn(1).item() + 2.0)

        ## inject anomaly
        node_feature_anomaly = None
        if block_an != "none" and feature_an != "none":
            node_feature_anomaly = False if node_edge_an == "edge_only" else True
            edge_feature_anomaly = False if node_edge_an == "node_only" else True

            if feature_an == "outside_ci":
                data_new = inject_dense_block_and_feature_anomaly(
                    data_new,
                    node_feature_anomaly,
                    edge_feature_anomaly,
                    num_group=1,
                    num_nodes_dict=num_nodes_dict,
                    connected_prop=connected_prop,
                    feature_anomaly_type="outside_ci",
                    prop_feat_node=prop_feat_node,
                    std_cutoff_node=std_cutoff_node,
                    prop_feat_edge=prop_feat_edge,
                    std_cutoff_edge=std_cutoff_edge,
                )

            elif feature_an == "scaled_gaussian":
                data_new = inject_dense_block_and_feature_anomaly(
                    data_new,
                    node_feature_anomaly,
                    edge_feature_anomaly,
                    num_group=1,
                    num_nodes_dict=num_nodes_dict,
                    connected_prop=connected_prop,
                    feature_anomaly_type="scaled_gaussian",
                    scale_node=scale_node,
                    scale_edge=scale_edge,
                )

        elif block_an != "none" and feature_an == "none":
            data_new = inject_dense_block_anomaly(
                data_new,
                num_group=1,
                num_nodes_dict=num_nodes_dict,
                connected_prop=connected_prop,
            )

        elif block_an == "none" and feature_an != "none":
            node_anomaly = False if node_edge_an == "edge_only" else True
            edge_anomaly = False if node_edge_an == "node_only" else True

            if feature_an == "outside_ci":
                data_new = inject_feature_anomaly(
                    data_new,
                    node_anomaly,
                    edge_anomaly,
                    feature_anomaly_type="outside_ci",
                    prop_feat_node=prop_feat_node,
                    std_cutoff_node=std_cutoff_node,
                    prop_feat_edge=prop_feat_edge,
                    std_cutoff_edge=std_cutoff_edge,
                )

            elif feature_an == "scaled_gaussian":
                data_new = inject_feature_anomaly(
                    data_new,
                    node_anomaly,
                    edge_anomaly,
                    feature_anomaly_type="scaled_gaussian",
                    scale_node=scale_node,
                    scale_edge=scale_edge,
                )

        list_affected_nodes = {
            nt: data_new[nt].y.sum().item() for nt in data_new.node_types
        }

        affected_nodes = sum(
            [data_new[nt].y.sum().item() for nt in data_new.node_types]
        )
        affected_edges = sum(
            [data_new[et].ye.sum().item() for et in data_new.edge_types]
        )
        print(
            f"affected: nodes = {affected_nodes}, edges = {affected_edges} | {list_affected_nodes} ",
            end="",
        )
        print(
            f"[{block_an}:{connected_prop:.2f},{feature_an},{num_nodes_dict.values()},{node_feature_anomaly}]"
        )

    return data_new
