import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv


class GraphSAINT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else hidden_channels
            self.convs.append(GraphConv(in_channels, hidden_channels))
        self.lin = torch.nn.Linear(num_layers * hidden_channels, out_channels)

    def set_aggr(self, aggr):
        self.conv1.aggr = aggr
        self.conv2.aggr = aggr
        self.conv3.aggr = aggr

    def forward(self, x, edge_index, edge_weight=None):
        xs = []
        for i, conv in enumerate(self.convs):
            x = F.relu(self.convs[i](x, edge_index, edge_weight))
            x = F.dropout(x, p=0.5, training=self.training)
            xs.append(x)
        x = torch.cat(xs, dim=-1)
        x = self.lin(x)
        return x.log_softmax(dim=-1)


    def inference(self, x_all, subgraph_loader, device):
        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:
        all_xs = []
        for i, conv in enumerate(self.convs):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id.to(x_all.device)].to(device)
                x = F.relu(self.convs[i](x, batch.edge_index.to(device)))
                x = F.dropout(x, p=0.5, training=self.training)
                xs.append(x[:batch.batch_size])
            x_all = torch.cat(xs, dim=0)
            all_xs.append(x_all)
        x = torch.cat(all_xs, dim=-1)
        embeddings = x.detach().clone()
        x = self.lin(x)
        return x, embeddings
