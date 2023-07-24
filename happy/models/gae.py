import torch
from torch_geometric.nn.conv import GCNConv
from torch_geometric.nn.pool import fps
from torch_geometric.nn.unpool import knn_interpolate

from happy.models.utils.custom_layers import KnnEdges


class GAE(torch.nn.Module):
    def __init__(
        self, in_channels, hidden_channels, depth, use_edge_weights, pool_method, pool_ratio=0.25
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.depth = depth
        self.use_edge_weights = use_edge_weights
        self.pool_method = pool_method
        self.pool_ratio = pool_ratio

        self.down_convs = torch.nn.ModuleList()
        self.knn_edge_transform = torch.nn.ModuleList()
        self.up_convs = torch.nn.ModuleList()
        self.lin = torch.nn.Linear(hidden_channels, hidden_channels)

        for i in range(depth):
            in_channels = in_channels if i == 0 else hidden_channels
            self.down_convs.append(GCNConv(in_channels, hidden_channels, improved=True))
            self.knn_edge_transform.append(
                KnnEdges(start_k=6, k_increment=1, no_op=False)
            )
        for i in range(depth):
            self.up_convs.append(
                GCNConv(hidden_channels, hidden_channels, improved=True)
            )

    def forward(self, batch):
        if not self.use_edge_weights:
            batch.edge_weights = None
        x = batch.x
        pos = batch.pos
        edge_index = batch.edge_index
        edge_weights = batch.edge_weights
        batch = batch.batch

        # todo: if this uses too much memory, we can save perm rather than pos
        # Save for reconstruction
        all_pos = [pos]
        edge_indices = [edge_index]
        all_edge_weights = [edge_weights]
        all_batch = [batch]

        # Downward pass
        for i in range(self.depth):
            x = self.down_convs[i](x, edge_index, edge_weights)
            x = torch.relu(x)
            if self.pool_method == "fps":
                perm = fps(pos, batch, ratio=self.pool_ratio)
            elif self.pool_method == "random":
                num_to_keep = int(x.shape[0] * self.pool_ratio)
                perm = torch.randperm(x.shape[0])[:num_to_keep]
            else:
                raise NotImplementedError
            batch = batch[perm]
            x, pos, edge_index, edge_weights, _, _, _ = self.knn_edge_transform[i](
                x, pos, edge_index, edge_weights, batch, perm, None, i
            )
            x = x[perm]
            all_pos.append(pos)
            edge_indices.append(edge_index)
            all_edge_weights.append(edge_weights)
            all_batch.append(batch)

        # Upward pass
        for i in range(self.depth):
            x = knn_interpolate(
                x,
                all_pos[-i - 1],
                all_pos[-i - 2],
                k=6,
                batch_x=all_batch[-i - 1],
                batch_y=all_batch[-i - 2],
            )
            x = self.up_convs[i](x, edge_indices[-i - 2], all_edge_weights[-i - 2])
            x = torch.relu(x)

        x = self.lin(x)
        return x

