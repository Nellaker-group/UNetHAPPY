import albumentations as al
from torchvision import transforms

from happy.data.transforms.agumentations import (
    AlbAugmenter,
    Stain_Augment_stylealb,
    GaussNoise_Augment_stylealb,
)
from happy.data.dataset.cell_dataset import CellDataset
from happy.data.transforms.transforms import Normalizer, Resizer


def get_cell_dataset(
    organ, split, annot_dir, dataset_names, image_size, oversampled
):
    augmentations = True if split == "train" else False
    transform = _setup_transforms(
        image_size, augmentations, padding=False, scale_annot=False
    )
    dataset = CellDataset(
        organ=organ,
        annotations_dir=annot_dir,
        dataset_names=dataset_names,
        split=split,
        oversampled_train=oversampled,
        transform=transform,
    )
    return dataset


# TODO: make this work for cells or nuclei. Right now it just works for cells
def _setup_transforms(image_size, augmentations=True, padding=True, scale_annot=True):
    transform = []
    if augmentations:
        transform.append(_setup_cell_augmentations())
    transform.append(Normalizer())
    transform.append(
        Resizer(
            min_side=image_size[0],
            max_side=image_size[1],
            padding=padding,
            scale_annotations=scale_annot,
        )
    )
    return transforms.Compose(transform)


def _setup_cell_augmentations():
    return AlbAugmenter(
        list_of_albumentations=[
            al.Flip(p=0.9),
            al.RandomRotate90(p=0.9),
            Stain_Augment_stylealb(p=0.9, variance=0.4),
            GaussNoise_Augment_stylealb(var_limit=(0.05, 0.2), p=0.2),
            GaussNoise_Augment_stylealb(var_limit=(0.01, 0.05), p=0.2),
            al.Blur(blur_limit=5, p=0.8),
            al.Rotate(limit=(0, 45), p=0.8),
        ],
        bboxes=False,
    )
