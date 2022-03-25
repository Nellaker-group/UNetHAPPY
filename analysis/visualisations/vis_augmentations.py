from pathlib import Path
import os
from collections import namedtuple

import typer
import cv2
import torch

from happy.data.transforms.transforms import untransform_image
from happy.train.nuc_train import setup_data
from happy.data.setup_data import setup_nuclei_datasets
from happy.data.setup_dataloader import setup_dataloaders


def main(
    project_name: str = typer.Option(...),
    annot_dir: str = typer.Option(...),
    dataset_name: str = typer.Option(...),
    num_images: int = 10,
):
    """Visualises the effect of augmentations on training data

    Args:
        project_name: name of the project dir to save visualisations to
        annot_dir: relative path to annotations
        dataset_name: the dataset who's validation set to evaluate over
        num_images: the number of images to evaluate
    """
    project_dir = (
        Path(__file__).absolute().parent.parent.parent / "projects" / project_name
    )
    os.chdir(str(project_dir))

    datasets = setup_nuclei_datasets(
        project_dir / annot_dir, dataset_name, False
    )
    dataloaders = setup_dataloaders(True, datasets, 1, 1, 1)


    HPs = namedtuple("HPs", "dataset_names batch")
    hp = HPs(dataset_name, 1)

    dataloaders = setup_data(project_dir / annot_dir, hp, False, 3, 1)

    with torch.no_grad():
        for idx, data in enumerate(dataloaders["train"]):
            if idx >= num_images:
                break

            img = untransform_image(data["img"][0])

            save_dir = (
                project_dir
                / "visualisations"
                / "nuclei"
                / "pred"
                / f"{dataset_name}_pred"
            )
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"val_{idx}.png"

            cv2.imwrite(str(save_path), img)


if __name__ == "__main__":
    typer.run(main)
