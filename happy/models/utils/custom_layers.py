from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn as nn
from torch import Tensor

from torch_geometric.nn.conv import SAGEConv
from torch_geometric.nn.pool import TopKPooling, knn_graph
from torch_geometric.transforms import Distance
from torch_geometric.data import Data
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.typing import Adj, OptPairTensor, Size, SparseTensor
from torch_geometric.utils import spmm
from tqdm import tqdm


class WeightedSAGEConv(SAGEConv):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        aggr: Optional[Union[str, List[str], Aggregation]] = "mean",
        normalize: bool = False,
        root_weight: bool = True,
        project: bool = False,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(
            in_channels,
            out_channels,
            aggr,
            normalize,
            root_weight,
            project,
            bias,
            **kwargs,
        )

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        edge_weight: Tensor = None,
        size: Size = None,
    ) -> Tensor:
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        if self.project and hasattr(self, "lin"):
            x = (self.lin(x[0]).relu(), x[1])

        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=size)
        out = self.lin_l(out)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            out = out + self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2.0, dim=-1)

        return out

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return x_j if edge_weight is None else x_j * edge_weight.view(-1, 1)

    def message_and_aggregate(
        self, adj_t: SparseTensor, x: OptPairTensor, edge_weight: Tensor = None
    ) -> Tensor:
        if isinstance(adj_t, SparseTensor):
            if edge_weight is not None:
                adj_t = adj_t.set_value(edge_weight, layout=None)
            else:
                adj_t = adj_t.set_value(None, layout=None)
        return spmm(adj_t, x[0], reduce=self.aggr)


class TopKPoolKnnEdges(TopKPooling):
    def __init__(self, in_channels, ratio, **kwargs):
        super().__init__(in_channels, ratio, **kwargs)

    def forward(self, x, edge_index, pos=None, edge_attr=None, batch=None, attn=None):
        x, edge_index, edge_attr, batch, perm, score = super().forward(
            x, edge_index, edge_attr, batch, attn
        )
        pos, edge_index, edge_attr = self._reconstruct_edges(
            pos, edge_attr, batch, perm
        )
        return x, pos, edge_index, edge_attr, batch, perm, score

    def _reconstruct_edges(self, pos, edge_attr, batch, perm):
        pos = pos[perm]
        edge_index = knn_graph(pos, k=6, batch=batch, loop=True)
        temp_data = Data(pos=pos, edge_index=edge_index, edge_attr=edge_attr)
        temp_data = Distance(cat=False, norm=True)(temp_data)
        edge_attr = temp_data.edge_attr
        return pos, edge_index, edge_attr


class KnnEdges(nn.Module):
    def __init__(self, start_k=6, k_increment=0, no_op=False):
        super().__init__()
        self.start_k = start_k
        self.k_increment = k_increment
        self.no_op = no_op

    def forward(self, x, pos, edge_index, edge_attr, batch, perm, score, i):
        if self.no_op:
            return x, pos, edge_index, edge_attr, batch, perm, score
        k = self.start_k + (self.k_increment * i)
        pos = pos[perm]
        edge_index = knn_graph(pos, k=k, batch=batch, loop=True)
        temp_data = Data(pos=pos, edge_index=edge_index, edge_attr=edge_attr)
        temp_data = Distance(cat=False, norm=True)(temp_data)
        if edge_attr is not None:
            edge_attr = temp_data.edge_attr
        return x, pos, edge_index, edge_attr, batch, perm, score


def pool_one_hop(edge_index, num_nodes, iteration_size):
    device = edge_index.device
    perm = []  # This will store our super nodes
    node_mask = torch.ones(num_nodes, dtype=torch.bool, device=device)

    while node_mask.any():  # While there are nodes left
        # Randomly select iteration_size nodes which are still available
        available_nodes = torch.where(node_mask)[0]
        if available_nodes.size(0) > iteration_size:
            supernodes = available_nodes[
                torch.randint(0, available_nodes.size(0), (iteration_size,))
            ]
        else:
            supernodes = available_nodes

        # Append supernodes to perm
        perm.extend(supernodes.tolist())

        # Remove the neighbors of supernodes from the selection pool
        rows, cols = edge_index
        neighbors = torch.unique(cols[torch.isin(rows, supernodes)])
        # Exclude supernodes from the neighbors list to account for self-loops
        neighbors = neighbors[~torch.isin(neighbors, supernodes)]

        node_mask[supernodes] = False
        node_mask[neighbors] = False

    return torch.tensor(perm, dtype=torch.long)
