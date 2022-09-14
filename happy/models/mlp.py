import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
import torch
from torch_geometric.nn import norm


class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.lins = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        self.bns.append(norm.BatchNorm(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(norm.BatchNorm(hidden_channels))
        
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

    def forward(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(self.bns[i](x))
            x = F.dropout(x, p=0.5, training=self.training)
        x = self.lins[-1](x)
        return torch.log_softmax(x, dim=-1)

    def inference(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(self.bns[i](x))
        x = self.lins[-1](x)
        return torch.log_softmax(x, dim=-1), x