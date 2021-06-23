import random

import torch
from torch.utils.data.sampler import Sampler
from torch_cluster import random_walk
from torch_geometric.data import NeighborSampler as BaseNeighborSampler


class AspectRatioBasedSampler(Sampler):

    def __init__(self, data_source, batch_size, drop_last):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.groups = self.group_images()

    def __iter__(self):
        random.shuffle(self.groups)
        for group in self.groups:
            yield group

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def group_images(self):
        # determine the order of the images
        order = list(range(len(self.data_source)))
        order.sort(key=lambda x: self.data_source.image_aspect_ratio(x))
        # NOTE: All aspect ratios are the same so shuffle the batches instead
        if min([self.data_source.image_aspect_ratio(x) for x in range(len(self.data_source))]) == max(
                [self.data_source.image_aspect_ratio(x) for x in range(len(self.data_source))]):
            random.shuffle(order)

        # divide into groups, one group = one batch
        return [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in
                range(0, len(order), self.batch_size)]


# class NeighborSampler(BaseNeighborSampler):
#     def sample(self, batch):
#         batch = torch.tensor(batch)
#         row, col, _ = self.adj_t.coo()
#
#         # For each node in `batch`, we sample a direct neighbor (as positive
#         # example) and a random node (as negative example):
#         pos_batch = random_walk(row, col, batch, walk_length=1,
#                                 coalesced=False)[:, 1]
#
#         neg_batch = torch.randint(0, self.adj_t.size(1), (batch.numel(), ),
#                                   dtype=torch.long)
#
#         batch = torch.cat([batch, pos_batch, neg_batch], dim=0)
#         return super(NeighborSampler, self).sample(batch)
