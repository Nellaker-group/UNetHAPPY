from pathlib import Path
from enum import Enum
import os

import typer
import pandas as pd
import cv2
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from happy.data.transforms.collaters import collater
from happy.data.dataset.nuclei_dataset import NucleiDataset
from happy.data.samplers.samplers import AspectRatioBasedSampler
from happy.data.transforms.transforms import Normalizer, Resizer, untransform_image
from happy.models import retinanet
from happy.utils.utils import load_weights, get_device
from happy.data.utils import draw_box, draw_centre
from happy.microscopefile.prediction_saver import PredictionSaver
from happy.organs.organs import get_organ


class ShapeArg(str, Enum):
    box = "box"
    point = "point"


def main(
    project_name: str = typer.Option(...),
    organ_name: str = typer.Option(...),
    annot_path: str = typer.Option(...),
    pre_trained: str = typer.Option(...),
    shape: ShapeArg = ShapeArg.point,
    dataset_name: str = typer.Option(...),
    score_threshold: float = 0.2,
    max_detections: int = 150,
    num_images: int = 10,
):
    """Visualises network predictions as boxes or points for one dataset

    Args:
        project_name: name of the project dir to save visualisations to
        organ_name: name of organ
        annot_path: relative path to annotations
        pre_trained: relative path to pretrained model
        shape: one of 'box' or 'point' for visualising the prediction
        dataset_name: the dataset who's validation set to evaluate over
        score_threshold: the confidence threshold below which to discard predictions
        max_detections: number of maximum detections to save, ordered by score
        num_images: the number of images to evaluate
    """
    device = get_device()

    organ = get_organ(organ_name)

    project_dir = (
        Path(__file__).absolute().parent.parent.parent / "projects" / project_name
    )
    annotation_path = project_dir / annot_path
    os.chdir(str(project_dir))

    dataset = NucleiDataset(
        annotations_dir=annotation_path,
        dataset_names=[dataset_name],
        split="val",
        transform=transforms.Compose([Normalizer(), Resizer()]),
    )
    print("Dataset configured")

    sampler = AspectRatioBasedSampler(dataset, batch_size=1, drop_last=False)
    dataloader = DataLoader(
        dataset,
        num_workers=3,
        collate_fn=collater,
        batch_sampler=sampler,
        shuffle=False,
    )
    print("Dataloaders configured")

    model = retinanet.build_retina_net(
        dataset.num_classes(), device=device, pretrained=False, resnet_depth=101
    )

    state_dict = torch.load(pre_trained, map_location=device)
    model = load_weights(state_dict, model)
    model = model.to(device)
    model.eval()
    print("Pushed model to device")

    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            if idx >= num_images:
                break

            scale = data["scale"]

            scores, _, boxes = model(data["img"].to(device).float(), device)
            scores = scores.cpu().numpy()
            boxes = boxes.cpu().numpy()
            boxes /= scale

            filtered_preds = PredictionSaver.filter_by_score(
                max_detections, score_threshold, scores, boxes
            )

            img = untransform_image(data["img"][0])

            save_dir = (
                project_dir / "visualisations" / "nuclei" / f"{dataset_name}_pred"
            )
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"val_{idx}.png"

            if len(filtered_preds) != 0:
                all_predictions = pd.DataFrame(
                    filtered_preds, columns=["x1", "y1", "x2", "y2"]
                )

                for i in range(len(all_predictions.index)):
                    x1 = all_predictions["x1"][i]
                    y1 = all_predictions["y1"][i]
                    x2 = all_predictions["x2"][i]
                    y2 = all_predictions["y2"][i]
                    label_name = "nucleus"

                    if shape.value == "point":
                        draw_centre(img, x1, y1, x2, y2, label_name, organ, cell=False)
                    elif shape.value == "box":
                        draw_box(img, x1, y1, x2, y2, label_name, organ, cell=False)
                    else:
                        raise ValueError(f"No such draw shape {shape.value}")

                print(f"saving to: {save_path}")
            else:
                print(f"no predictions on val_{idx}.png")

            cv2.imwrite(str(save_path), img)


if __name__ == "__main__":
    typer.run(main)
