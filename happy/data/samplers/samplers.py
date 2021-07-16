import random

from torch.utils.data.sampler import Sampler


class AspectRatioBasedSampler(Sampler):
    def __init__(self, dataset, batch_size, drop_last):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.groups = self.group_images()

    def __iter__(self):
        random.shuffle(self.groups)
        for group in self.groups:
            yield group

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def group_images(self):
        # determine the order of the images
        order = list(range(len(self.dataset)))
        order.sort(key=lambda x: self.dataset.image_aspect_ratio(x))
        # NOTE: All aspect ratios are the same so shuffle the batches instead
        if min(
            [self.dataset.image_aspect_ratio(x) for x in range(len(self.dataset))]
        ) == max(
            [self.dataset.image_aspect_ratio(x) for x in range(len(self.dataset))]
        ):
            random.shuffle(order)

        # divide into groups, one group = one batch
        return [
            [order[x % len(order)] for x in range(i, i + self.batch_size)]
            for i in range(0, len(order), self.batch_size)
        ]
