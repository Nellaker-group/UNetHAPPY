from torch.utils.data.sampler import BatchSampler, WeightedRandomSampler
from torch.utils.data import DataLoader

from happy.data.samplers.samplers import AspectRatioBasedSampler
from happy.data.transforms.collaters import cell_collater, collater


def setup_dataloaders(nuclei, datasets, num_workers, batch_size):
    collate_fn = collater if nuclei else cell_collater

    dataloaders = {}
    for dataset in datasets:
        if dataset == "train":
            dataloaders[dataset] = get_dataloader(
                "train",
                datasets[dataset],
                collate_fn,
                num_workers,
                nuclei,
                batch_size,
            )
        else:
            dataloaders[dataset] = get_dataloader(
                "val", datasets[dataset], collate_fn, num_workers, nuclei, batch_size
            )
    print("Dataloaders configured")
    return dataloaders


def get_dataloader(
    split, dataset, collater, num_workers, nuclei, train_batch_size=None
):
    batch_size = train_batch_size if split == "train" else 1
    if split == "train" and nuclei == False:
        sampler = BatchSampler(
            WeightedRandomSampler(
                dataset.class_sampling_weights, len(dataset), replacement=True
            ),
            batch_size=batch_size,
            drop_last=False,
        )
    else:
        sampler = AspectRatioBasedSampler(
            dataset, batch_size=batch_size, drop_last=False
        )
    return DataLoader(
        dataset, num_workers=num_workers, collate_fn=collater, batch_sampler=sampler
    )
