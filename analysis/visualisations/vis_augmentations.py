from pathlib import Path
import os

import typer
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as al
from tqdm import tqdm
from torchvision import transforms

from happy.data.transforms.agumentations import AlbAugmenter, StainAugment
from happy.data.transforms.transforms import Normalizer, Resizer, unnormalise_image
from happy.data.transforms.utils.color_conversion import get_rgb_matrices
from happy.data.dataset.nuclei_dataset import NucleiDataset
from happy.data.setup_dataloader import get_dataloader
from happy.data.transforms.collaters import collater
from happy.data.utils import draw_centre


def main(
    project_name: str = typer.Option(...),
    annot_dir: str = typer.Option(...),
    split: str = typer.Option(...),
    dataset_name: str = typer.Option(...),
    num_images: int = 10,
    plot_groundtruth: bool = True,
):
    """Visualises the effect of augmentations across nuclei dataset

    Args:
        project_name: name of the project dir to save visualisations to
        annot_dir: relative path to annotations
        split: the dataset split to run over
        dataset_name: the dataset who's validation set to evaluate over
        num_images: the number of images to evaluate
        plot_groundtruth: whether to overlay groundtruth points on image after aug
    """
    project_dir = (
        Path(__file__).absolute().parent.parent.parent / "projects" / project_name
    )
    os.chdir(str(project_dir))

    transform = transforms.Compose(
        [_get_transformations(), Normalizer(), Resizer(padding=False)]
    )
    dataset = NucleiDataset(
        annotations_dir=project_dir / annot_dir,
        dataset_names=[dataset_name],
        split=split,
        transform=transform,
    )
    dataloader = get_dataloader(split, dataset, collater, 1, True, 1)

    with torch.no_grad():
        for idx, data in enumerate(tqdm(dataloader)):
            if idx >= num_images:
                break

            img = data["img"][0]
            img = unnormalise_image(img)
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            if plot_groundtruth:
                gt_predictions = pd.DataFrame(
                    data["annot"].numpy()[0][:, :4], columns=["x1", "y1", "x2", "y2"]
                )
                for i in range(len(gt_predictions.index)):
                    x1 = gt_predictions["x1"][i]
                    y1 = gt_predictions["y1"][i]
                    x2 = gt_predictions["x2"][i]
                    y2 = gt_predictions["y2"][i]
                    label = "nucleus"
                    draw_centre(img, x1, y1, x2, y2, label, None, False, (0, 255, 255))

            save_dir = (
                project_dir
                / "visualisations"
                / "nuclei"
                / "augmentations"
                / f"{dataset_name}_aug"
            )
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"{idx}.png"

            cv2.imwrite(str(save_path), img)


def _get_transformations():
    alb = [
        al.Flip(p=0.5),
        al.RandomRotate90(p=0.5),
        StainAugment(get_rgb_matrices(), p=0.9, variance=0.4),
        al.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.8),
        al.GaussNoise(var_limit=(10.0, 200.0), p=0.8),
        al.Blur(blur_limit=5, p=0.8),
    ]
    return AlbAugmenter(list_of_albumentations=alb, bboxes=True)


if __name__ == "__main__":
    typer.run(main)
