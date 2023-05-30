import torch
from torch import nn as nn
from torch.nn import functional as F
from torch_geometric.nn import MLP, GINEConv, GINConv


class ClusterGIN(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        dropout,
        num_layers,
        reduce_dims=None,
        include_edge_attr=False,
    ):
        super(ClusterGIN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.reduce_dims = reduce_dims
        self.include_edge_attr = include_edge_attr
        self.convs = nn.ModuleList()

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else hidden_channels
            if i == num_layers - 1 and reduce_dims is None:
                hidden_channels = out_channels
            else:
                hidden_channels = hidden_channels
            mlp = MLP([in_channels, hidden_channels, hidden_channels])
            if self.include_edge_attr:
                self.convs.append(GINEConv(mlp, train_eps=False, edge_dim=1))
            else:
                self.convs.append(GINConv(mlp, train_eps=False))

        if reduce_dims is not None:
            self.lin1 = nn.Linear(hidden_channels, reduce_dims)
            self.lin2 = nn.Linear(reduce_dims, out_channels)

    def forward(self, x, edge_index, edge_attr):
        for i, conv in enumerate(self.convs):
            if self.include_edge_attr:
                x = conv(x, edge_index, edge_attr)
            else:
                x = conv(x, edge_index)
            if i == len(self.convs) - 1 and self.reduce_dims is None:
                continue
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        if self.reduce_dims is not None:
            x = self.lin1(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def inference(self, x_all, edge_attr, subgraph_loader, device):
        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        if self.include_edge_attr:
            edge_attr = edge_attr.to(device)
        for i, conv in enumerate(self.convs):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, e_id, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                if self.include_edge_attr:
                    edge_attr_batch = edge_attr[e_id].to(device)
                    x = conv((x, x_target), edge_index, edge_attr_batch)
                else:
                    x = conv((x, x_target), edge_index)
                if i != len(self.convs) - 1 or self.reduce_dims is not None:
                    x = F.relu(x)
                xs.append(x)
            x_all = torch.cat(xs, dim=0)

            if i == self.num_layers - 2 and self.reduce_dims is None:
                embeddings = x_all.detach().cpu().clone()

        if self.reduce_dims is not None:
            x_all = self.lin1(x_all)
            x_all = F.relu(x_all)
            embeddings = x_all.detach().cpu().clone()
            x_all = self.lin2(x_all)

        return x_all.cpu(), embeddings
