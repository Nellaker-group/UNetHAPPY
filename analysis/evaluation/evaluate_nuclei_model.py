from pathlib import Path
from collections import namedtuple
from typing import List
import os

import typer
import torch
import numpy as np
from tqdm import tqdm

from happy.utils.utils import get_device
from happy.microscopefile.prediction_saver import PredictionSaver
from happy.train.calc_point_eval import (
    convert_boxes_to_points,
    evaluate_points_in_image,
)
from happy.train.nuc_train import setup_data, setup_model


def main(
    project_name: str = typer.Option(...),
    annot_dir: str = typer.Option(...),
    pre_trained: str = typer.Option(...),
    dataset_names: List[str] = typer.Option([]),
    score_threshold: float = 0.4,
    max_detections: int = 500,
    valid_point_range: int = 30,
):
    """Visualises network predictions as boxes or points for one dataset

    Args:
        project_name: name of the project dir to save visualisations to
        annot_dir: relative path to annotations
        pre_trained: relative path to pretrained model
        dataset_names: the datasets who's validation set to evaluate over
        score_threshold: the confidence threshold below which to discard predictions
        max_detections: number of maximum detections to save, ordered by score
        valid_point_range: distance to gt in pixels for which a prediction is valid
    """
    device = get_device()

    project_dir = (
        Path(__file__).absolute().parent.parent.parent / "projects" / project_name
    )
    os.chdir(str(project_dir))

    HPs = namedtuple("HPs", "dataset_names batch")
    hp = HPs(dataset_names, 1)

    multiple_val_sets = True if len(dataset_names) > 1 else False
    dataloaders = setup_data(project_dir / annot_dir, hp, multiple_val_sets, 3, 1)
    dataloaders.pop("train")
    model = setup_model(False, device, False, pre_trained)
    model.eval()

    mean_precision = {}
    mean_recall = {}
    mean_f1 = {}
    with torch.no_grad():
        for dataset_name in dataloaders:
            all_precision = []
            all_recall = []
            all_f1 = []
            for data in tqdm(dataloaders[dataset_name]):
                scale = data["scale"]

                scores, _, boxes = model(data["img"].to(device).float(), device)
                scores = scores.cpu().numpy()
                boxes = boxes.cpu().numpy()
                boxes /= scale

                filtered_preds = PredictionSaver.filter_by_score(
                    max_detections, score_threshold, scores, boxes
                )
                gt_predictions = data["annot"].numpy()[0][:, :4]
                gt_predictions /= scale[0]

                ground_truth_points = convert_boxes_to_points(gt_predictions)
                predicted_points = convert_boxes_to_points(filtered_preds)
                precision, recall, f1 = evaluate_points_in_image(
                    ground_truth_points, predicted_points, valid_point_range
                )
                all_precision.append(precision)
                all_recall.append(recall)
                all_f1.append(f1)
            mean_precision[dataset_name] = np.mean(np.array(all_precision)).round(4)
            mean_recall[dataset_name] = np.mean(np.array(all_recall)).round(4)
            mean_f1[dataset_name] = np.mean(np.array(all_f1)).round(4)

    print(f"Precision: {mean_precision}")
    print(f"Recall: {mean_recall}")
    print(f"F1: {mean_f1}")

if __name__ == "__main__":
    typer.run(main)
