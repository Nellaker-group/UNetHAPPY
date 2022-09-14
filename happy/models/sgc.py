import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SGConv


class SGC(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.conv = SGConv(in_channels, out_channels, num_layers)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        return F.log_softmax(x, dim=1)