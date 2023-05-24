from typing import List, Optional, Tuple, Union
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn.conv import SAGEConv
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.typing import Adj, OptPairTensor, Size, SparseTensor
from torch_geometric.utils import spmm


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
        super().__init__(in_channels, out_channels, aggr, normalize, root_weight,
                         project, bias, **kwargs)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_weight: Tensor = None, size: Size = None) -> Tensor:
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        if self.project and hasattr(self, 'lin'):
            x = (self.lin(x[0]).relu(), x[1])

        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=size)
        out = self.lin_l(out)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            out = out + self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return x_j if edge_weight is None else x_j * edge_weight.view(-1, 1)

    def message_and_aggregate(self, adj_t: SparseTensor, x: OptPairTensor,
                              edge_weight: Tensor = None) -> Tensor:
        if isinstance(adj_t, SparseTensor):
            if edge_weight is not None:
                adj_t = adj_t.set_value(edge_weight, layout=None)
            else:
                adj_t = adj_t.set_value(None, layout=None)
        return spmm(adj_t, x[0], reduce=self.aggr)
