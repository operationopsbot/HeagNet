# Copyright 2024 Grabtaxi Holdings Pte Ltd (GRAB), All rights reserved.
# Use of this source code is governed by an MIT-style license that can be found in the LICENSE file

from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import math

from torch_geometric.data import Data, FeatureStore, GraphStore, HeteroData
from torch_geometric.loader.neighbor_loader import NeighborLoader
from torch_geometric.loader.node_loader import NodeLoader
from torch_geometric.sampler import NeighborSampler
from torch_geometric.typing import EdgeType, InputNodes, OptTensor, NodeType


class MultiNeighborLoader:
    def __init__(
        self,
        data: Union[Data, HeteroData, Tuple[FeatureStore, GraphStore]],
        num_neighbors: Union[List[int], Dict[EdgeType, List[int]]],
        input_nodes: List[NodeType],
        batch_size: Union[List[int], int],
        input_repeat: Union[List[int], List[float], None] = None,
        input_time: OptTensor = None,
        replace: bool = False,
        directed: bool = True,
        disjoint: bool = False,
        temporal_strategy: str = "uniform",
        time_attr: Optional[str] = None,
        transform: Optional[Callable] = None,
        transform_sampler_output: Optional[Callable] = None,
        is_sorted: bool = False,
        filter_per_worker: bool = False,
        neighbor_sampler: Optional[NeighborSampler] = None,
        **kwargs,
    ):
        if input_time is not None and time_attr is None:
            raise ValueError(
                "Received conflicting 'input_time' and "
                "'time_attr' arguments: 'input_time' is set "
                "while 'time_attr' is not set."
            )

        if neighbor_sampler is None:
            neighbor_sampler = NeighborSampler(
                data,
                num_neighbors=num_neighbors,
                replace=replace,
                directed=directed,
                disjoint=disjoint,
                temporal_strategy=temporal_strategy,
                time_attr=time_attr,
                is_sorted=is_sorted,
                share_memory=kwargs.get("num_workers", 0) > 0,
            )

        loaders = []
        len_all = 0
        id_flag = []

        if input_repeat is None:
            input_repeat = [1 for _ in input_nodes]

        if isinstance(batch_size, int):
            batch_size = [batch_size for _ in input_nodes]

        for i, nt in enumerate(input_nodes):
            n_node = data[nt].num_nodes
            if input_repeat[i] != 1:
                ns = int(math.ceil(input_repeat[i] * n_node))
                rid = torch.randint(low=0, high=n_node, size=(ns,))
            else:
                rid = torch.arange(n_node)

            ld = NodeLoader(
                data=data,
                node_sampler=neighbor_sampler,
                input_nodes=(nt, rid),
                batch_size=batch_size[i],
                input_time=input_time,
                transform=transform,
                transform_sampler_output=transform_sampler_output,
                filter_per_worker=filter_per_worker,
                **kwargs,
            )
            nd = len(ld)

            id_flag.append(torch.ones(nd, dtype=torch.long) * i)

            len_all += nd
            loaders.append(ld)

            print(f"i: {i}, {nt}: nd {nd}, len_all {len_all}")

        self.loaders = loaders
        self.len_all = len_all

        perm = torch.randperm(len_all)
        flag_all = torch.cat(id_flag, dim=0)
        self.flag_perm = flag_all[perm]

        self.current = -1

    def __len__(self):
        return self.len_all

    def __iter__(self):
        self.iters = []
        for ld in self.loaders:
            self.iters.append(iter(ld))

        self.current = -1

        return self

    def __next__(self):
        self.current += 1
        if self.current < self.len_all:
            return next(self.iters[self.flag_perm[self.current]])

        raise StopIteration
