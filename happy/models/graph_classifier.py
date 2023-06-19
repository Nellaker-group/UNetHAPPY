import torch
from torch import nn as nn
from torch.nn import functional as F
from torch_geometric.nn.pool import TopKPooling, SAGPooling, EdgePooling, ASAPooling
from torch_geometric.nn.conv import GraphConv
from torch_geometric.nn import JumpingKnowledge
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_mean_pool as gap

from happy.models.utils.custom_layers import KnnEdges


class TopKClassifer(nn.Module):
    def __init__(
        self, in_channels, hidden_channels, out_channels, num_layers, pool_ratio
    ):
        super().__init__()
        self.num_layers = num_layers
        self.knn_edge_transform = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else hidden_channels
            self.convs.append(GraphConv(in_channels, hidden_channels))
            self.pools.append(TopKPooling(hidden_channels, ratio=pool_ratio))
            self.knn_edge_transform.append(
                KnnEdges(start_k=6, k_increment=2, no_op=False)
            )

        self.lin1 = nn.Linear(hidden_channels * 2, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, int(hidden_channels / 2))
        self.lin3 = nn.Linear(int(hidden_channels / 2), out_channels)

    def forward(self, data):
        x, pos, edge_index, edge_attr, batch = (
            data.x,
            data.pos,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        xs = []
        for i, conv in enumerate(self.convs):
            x = F.relu(conv(x, edge_index))
            x, edge_index, _, batch, perm, score = self.pools[i](
                x, edge_index, None, batch
            )
            xs.append(torch.cat([gmp(x, batch), gap(x, batch)], dim=1))
            _, pos, edge_index, edge_attr, _, _, _ = self.knn_edge_transform[i](
                x, pos, edge_index, edge_attr, batch, perm, score, i
            )

        x = torch.stack(xs).sum(dim=0)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = self.lin3(x)

        return x

    def forward_old(self, data):
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


class SAGClassifer(nn.Module):
    def __init__(
        self, in_channels, hidden_channels, out_channels, num_layers, pool_ratio
    ):
        super().__init__()
        self.num_layers = num_layers
        self.knn_edge_transform = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else hidden_channels
            self.convs.append(GraphConv(in_channels, hidden_channels, aggr="mean"))
            self.pools.append(SAGPooling(hidden_channels, ratio=pool_ratio))
            self.knn_edge_transform.append(
                KnnEdges(start_k=6, k_increment=2, no_op=False)
            )

        self.jump = JumpingKnowledge(mode="cat")
        self.lin1 = nn.Linear(hidden_channels * num_layers, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, pos, edge_index, edge_attr, batch = (
            data.x,
            data.pos,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        xs = []
        for i, conv in enumerate(self.convs):
            x = F.relu(conv(x, edge_index))
            x, edge_index, _, batch, perm, score = self.pools[i](
                x, edge_index, None, batch
            )
            xs += [gmp(x, batch)]
            _, pos, edge_index, edge_attr, _, _, _ = self.knn_edge_transform[i](
                x, pos, edge_index, edge_attr, batch, perm, score, i
            )

        x = self.jump(xs)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        return x


class ASAPClassifer(nn.Module):
    def __init__(
        self, in_channels, hidden_channels, out_channels, num_layers, pool_ratio
    ):
        super().__init__()
        self.num_layers = num_layers
        self.knn_edge_transform = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else hidden_channels
            self.convs.append(GraphConv(in_channels, hidden_channels, aggr="mean"))
            self.pools.append(ASAPooling(hidden_channels, ratio=pool_ratio))
            self.knn_edge_transform.append(
                KnnEdges(start_k=6, k_increment=2, no_op=False)
            )

        self.jump = JumpingKnowledge(mode="cat")
        self.lin1 = nn.Linear(hidden_channels * num_layers, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        score = None
        x, pos, edge_index, edge_attr, batch = (
            data.x,
            data.pos,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        xs = []
        for i, conv in enumerate(self.convs):
            x = F.relu(conv(x, edge_index))
            x, edge_index, _, batch, perm = self.pools[i](
                x, edge_index, None, batch
            )
            xs += [gmp(x, batch)]
            _, pos, edge_index, edge_attr, _, _, _ = self.knn_edge_transform[i](
                x, pos, edge_index, edge_attr, batch, perm, score, i
            )

        x = self.jump(xs)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        return x


# class EdgePoolClassifer(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
#         super().__init__()
#         self.num_layers = num_layers
#         self.knn_edge_transform = nn.ModuleList()
#         self.convs = nn.ModuleList()
#         self.pools = nn.ModuleList()
#
#         for i in range(num_layers):
#             in_channels = in_channels if i == 0 else hidden_channels
#             self.convs.append(GraphConv(in_channels, hidden_channels))
#             self.pools.append(EdgePooling(hidden_channels, dropout=0.2))
#
#         self.lin1 = nn.Linear(hidden_channels * 2, hidden_channels)
#         self.lin2 = nn.Linear(hidden_channels, int(hidden_channels / 2))
#         self.lin3 = nn.Linear(int(hidden_channels / 2), out_channels)
#
#     def forward(self, data):
#         x, pos, edge_index, edge_attr, batch = (
#             data.x,
#             data.pos,
#             data.edge_index,
#             data.edge_attr,
#             data.batch,
#         )
#
#         xs = []
#         for i, conv in enumerate(self.convs):
#             x = F.relu(conv(x, edge_index))
#             x, _, _, batch, perm, score = self.pools[i](x, edge_index, None, batch)
#             xs.append(torch.cat([gmp(x, batch), gap(x, batch)], dim=1))
#             # _, pos, edge_index, edge_attr, _, _, _ = self.knn_edge_transform[i](
#             #     x, pos, edge_attr, batch, perm, score, i
#             # )
#
#         x = torch.stack(xs).sum(dim=0)
#
#         x = F.relu(self.lin1(x))
#         x = F.dropout(x, p=0.5, training=self.training)
#         x = F.relu(self.lin2(x))
#         x = self.lin3(x)
#
#         return x
