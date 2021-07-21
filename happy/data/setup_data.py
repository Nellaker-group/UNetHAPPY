import albumentations as al
from torchvision import transforms

from happy.data.transforms.agumentations import (
    AlbAugmenter,
    Stain_Augment_stylealb,
    GaussNoise_Augment_stylealb,
)
from happy.data.dataset.cell_dataset import CellDataset
from happy.data.dataset.nuclei_dataset import NucleiDataset
from happy.data.transforms.transforms import Normalizer, Resizer


def setup_nuclei_datasets(annot_dir, dataset_names, multiple_val_sets):
    # Create the datasets from all directories specified in dataset_names
    dataset_train = get_nuclei_dataset("train", annot_dir, dataset_names)
    dataset_val = get_nuclei_dataset("val", annot_dir, dataset_names)
    datasets = {"train": dataset_train, "val_all": dataset_val}
    # Create validation datasets from all directories specified in dataset_names
    dataset_val_dict = {}
    if multiple_val_sets:
        for dataset_name in dataset_names:
            dataset_val_dict[dataset_name] = get_nuclei_dataset(
                "val", annot_dir, dataset_name
            )
        datasets.update(dataset_val_dict)
    print("Dataset configured")
    return datasets


def setup_cell_datasets(
    organ, annot_dir, dataset_names, image_size, multiple_val_sets, oversampled
):
    # Create the datasets from all directories specified in dataset_names
    dataset_train = get_cell_dataset(
        organ, "train", annot_dir, dataset_names, image_size, oversampled
    )
    dataset_val = get_cell_dataset(
        organ, "val", annot_dir, dataset_names, image_size, False
    )
    datasets = {"train": dataset_train, "val_all": dataset_val}
    # Create validation datasets from all directories specified in dataset_names
    dataset_val_dict = {}
    if multiple_val_sets:
        for dataset_name in dataset_names:
            dataset_val_dict[dataset_name] = get_cell_dataset(
                organ, "val", annot_dir, dataset_name, image_size, False
            )
        datasets.update(dataset_val_dict)
    print("Dataset configured")
    return datasets


def get_nuclei_dataset(split, annot_dir, dataset_names):
    augmentations = True if split == "train" else False
    transform = _setup_transforms(augmentations, nuclei=True)
    dataset = NucleiDataset(
        annotations_dir=annot_dir,
        dataset_names=dataset_names,
        split=split,
        transform=transform,
    )
    return dataset


def get_cell_dataset(organ, split, annot_dir, dataset_names, image_size, oversampled):
    augmentations = True if split == "train" else False
    transform = _setup_transforms(augmentations, image_size=image_size, nuclei=False)
    dataset = CellDataset(
        organ=organ,
        annotations_dir=annot_dir,
        dataset_names=dataset_names,
        split=split,
        oversampled_train=oversampled,
        transform=transform,
    )
    return dataset


def _setup_transforms(augmentations, image_size=None, nuclei=True):
    transform = []
    if augmentations:
        transform.append(_augmentations(nuclei))
    transform.append(Normalizer())
    if nuclei:
        transform.append(Resizer())
    else:
        transform.append(
            Resizer(
                min_side=image_size[0],
                max_side=image_size[1],
                padding=False,
                scale_annotations=False,
            )
        )
    return transforms.Compose(transform)


def _augmentations(nuclei):
    alb = [
        al.Flip(p=0.9),
        al.RandomRotate90(p=0.9),
        Stain_Augment_stylealb(p=0.9, variance=0.4),
        al.Blur(blur_limit=5, p=0.8),
        al.Rotate(limit=(0, 45), p=0.8),
    ]
    if nuclei:
        alb.insert(3, GaussNoise_Augment_stylealb(var_limit=(0.05, 0.2), p=0.85))
        alb.insert(4, GaussNoise_Augment_stylealb(var_limit=(0.01, 0.05), p=0.85))
        alb.insert(5, GaussNoise_Augment_stylealb(var_limit=(0.01, 0.05), p=0.85))
        alb.append(al.RandomScale(scale_limit=0.05, p=0.8))
        return AlbAugmenter(list_of_albumentations=alb)
    else:
        alb.insert(3, GaussNoise_Augment_stylealb(var_limit=(0.05, 0.2), p=0.2))
        alb.insert(4, GaussNoise_Augment_stylealb(var_limit=(0.01, 0.05), p=0.2))
        return AlbAugmenter(list_of_albumentations=alb, bboxes=False)
