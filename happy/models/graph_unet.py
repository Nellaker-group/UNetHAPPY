import torch
from torch_geometric.utils import add_self_loops, remove_self_loops
from torch_geometric.nn import GraphUNet


class UNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, depth, pool_ratio):
        super(UNet, self).__init__()
        self.conv = GraphUNet(
            in_channels, hidden_channels, in_channels, depth, pool_ratio)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        d, latent_x, latent_edge, batch = self.conv(x, edge_index, batch)
        d = d.to(self.device)
        return d, latent_x, latent_edge, batch
