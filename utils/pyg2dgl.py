# Copyright 2024 Grabtaxi Holdings Pte Ltd (GRAB), All rights reserved.
# Use of this source code is governed by an MIT-style license that can be found in the LICENSE file

import dgl
import torch
import torch_geometric
from torch_geometric.data import HeteroData


def hetero_pyg2dgl(
    data,
    pyg_nfeat="x",
    dgl_nfeat="feat",
    pyg_efeat="edge_attr",
    dgl_efeat="feat",
    pyg_label="y",
    dgl_label="label",
    pyg_adj="edge_index",
    include_data=True,
):
    dgl_graph_data = {}
    for etype in data.edge_types:
        dgl_graph_data[etype] = (
            data[etype][pyg_adj][0],
            data[etype][pyg_adj][1],
        )

    g = dgl.heterograph(dgl_graph_data)

    for ntype in data.node_types:
        for key in data[ntype]:
            if key == pyg_nfeat:
                g.nodes[ntype].data[dgl_nfeat] = data[ntype][pyg_nfeat]
            elif key == pyg_label:
                g.nodes[ntype].data[dgl_label] = data[ntype][pyg_label]
            else:
                if include_data:
                    if isinstance(data[ntype][key], torch.Tensor):
                        g.nodes[ntype].data[key] = data[ntype][key]

    for etype in data.edge_types:
        for key in data[etype]:
            if key == pyg_efeat:
                g.edges[etype].data[dgl_efeat] = data[etype][pyg_efeat]
            elif key == pyg_label:
                g.edges[etype].data[dgl_label] = data[etype][pyg_label]
            else:
                if include_data:
                    if key != pyg_adj and isinstance(data[etype][key], torch.Tensor):
                        g.edges[etype].data[key] = data[etype][key]

    return g


def hetero_dgl2pyg(
    g,
    pyg_nfeat="x",
    dgl_nfeat="feat",
    pyg_efeat="edge_attr",
    dgl_efeat="feat",
    pyg_label="y",
    dgl_label="label",
    pyg_adj="edge_index",
    include_data=True,
):

    data = HeteroData()
    for ntype in g.ntypes:
        for key in g.nodes[ntype].data:
            if key == dgl_nfeat:
                data[ntype][pyg_nfeat] = g.nodes[ntype].data[dgl_nfeat]
            elif key == dgl_label:
                data[ntype][pyg_label] = g.nodes[ntype].data[dgl_label]
            else:
                if include_data:
                    if isinstance(g.nodes[ntype].data[key], torch.Tensor):
                        data[ntype][key] = g.nodes[ntype].data[key]

    for etype in g.canonical_etypes:
        data[etype][pyg_adj] = torch.stack(g.edges(etype=etype))
        for key in g.edges[etype].data:
            if key == dgl_efeat:
                data[etype][pyg_efeat] = g.edges[etype].data[dgl_efeat]
            elif key == dgl_label:
                data[etype][pyg_label] = g.edges[etype].data[dgl_label]
            else:
                if include_data:
                    if isinstance(g.edges[etype].data[key], torch.Tensor):
                        data[etype][key] = g.edges[etype].data[key]

    return data


def pyg_structure_only(g, pyg_adj="edge_index"):
    data = HeteroData()

    for etype in g.edge_types:
        data[etype][pyg_adj] = g[etype][pyg_adj]

    return data


def dgl_hetero_to_bidirected(g):

    dgl_graph_data = {}
    for etype in g.canonical_etypes:
        edge_idx = g.edges(etype=etype)
        dgl_graph_data[etype] = edge_idx

        src, rel, dst = etype
        rev_etype = (dst, "rev_" + rel, src)
        dgl_graph_data[rev_etype] = (edge_idx[1], edge_idx[0])

    graph = dgl.heterograph(dgl_graph_data)

    for ntype in graph.ntypes:
        graph.nodes[ntype].data["_OID"] = torch.arange(graph.num_nodes(ntype))

    for etype in g.canonical_etypes:
        graph.edges[etype].data["_OID"] = torch.arange(graph.num_edges(etype))

        src, rel, dst = etype
        rev_etype = (dst, "rev_" + rel, src)
        graph.edges[rev_etype].data["_OID"] = torch.arange(graph.num_edges(etype))

    return graph


def pyg_stardard_names(g, pyg_adj="edge_index"):
    data = HeteroData()

    for etype in g.edge_types:
        data[etype][pyg_adj] = g[etype][pyg_adj]

    return data
