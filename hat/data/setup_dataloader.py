from torch.utils.data.sampler import BatchSampler, WeightedRandomSampler

from nucnet.data.samplers.samplers import AspectRatioBasedSampler
from torch.utils.data import DataLoader


def get_cell_dataloader(split, dataset, collator, train_batch_size=None):
    if split == "train":
        sampler = BatchSampler(
            WeightedRandomSampler(
                dataset.class_sampling_weights, len(dataset), replacement=True
            ),
            batch_size=train_batch_size,
            drop_last=False,
        )
    else:
        sampler = AspectRatioBasedSampler(dataset, batch_size=1, drop_last=False)

    return DataLoader(
        dataset, num_workers=10, collate_fn=collator, batch_sampler=sampler
    )
