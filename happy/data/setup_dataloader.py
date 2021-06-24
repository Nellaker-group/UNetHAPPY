from torch.utils.data.sampler import BatchSampler, WeightedRandomSampler
from torch.utils.data import DataLoader

from happy.data.samplers.samplers import AspectRatioBasedSampler
from happy.data.transforms.collaters import cell_collater


def setup_cell_dataloaders(datasets, batch_size):
    dataloaders = {}
    for dataset in datasets:
        if dataset == "train":
            dataloaders[dataset] = _get_cell_dataloader(
                "train", datasets[dataset], cell_collater, batch_size
            )
        else:
            dataloaders[dataset] = _get_cell_dataloader(
                "val", datasets[dataset], cell_collater
            )
    print("Dataloaders configured")
    return dataloaders


def _get_cell_dataloader(split, dataset, collator, train_batch_size=None):
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
