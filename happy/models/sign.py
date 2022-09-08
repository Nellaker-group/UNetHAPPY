import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
import torch


class SIGN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(SIGN, self).__init__()
        self.num_layers = num_layers

        self.lins = torch.nn.ModuleList()
        for _ in range(num_layers + 1):
            self.lins.append(Linear(in_channels, hidden_channels))
        self.lin = Linear((num_layers + 1) * hidden_channels, out_channels)

    def forward(self, xs):
        hs = []
        for x, lin in zip(xs, self.lins):
            h = lin(x).relu()
            h = F.dropout(h, p=0.5, training=self.training)
            hs.append(h)
        h = torch.cat(hs, dim=-1)
        o = self.lin(h)
        return o.log_softmax(dim=-1)

    def inference(self, xs):
        hs = []
        for x, lin in zip(xs, self.lins):
            h = lin(x).relu()
            h = F.dropout(h, p=0.5, training=self.training)
            hs.append(h)
        h = torch.cat(hs, dim=-1)
        o = self.lin(h)
        return o, h
