import torch
from torch import nn as nn
from torch.nn import functional as F
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_mean_pool as gap


class TopKClassifer(nn.Module):
    def __init__(
        self, in_channels, hidden_channels, out_channels, num_layers, pool_ratio
    ):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else hidden_channels
            self.convs.append(GraphConv(in_channels, hidden_channels))
            self.pools.append(TopKPooling(hidden_channels, ratio=pool_ratio))

        self.lin1 = nn.Linear(hidden_channels * 2, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, int(hidden_channels / 2))
        self.lin3 = nn.Linear(int(hidden_channels / 2), out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        xs = []
        for i, conv in enumerate(self.convs):
            x = F.relu(conv(x, edge_index))
            x, edge_index, _, batch, _, _ = self.pools[i](x, edge_index, None, batch)
            xs.append(torch.cat([gmp(x, batch), gap(x, batch)], dim=1))
        x = torch.stack(xs).sum(dim=0)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = self.lin3(x)

        return x
