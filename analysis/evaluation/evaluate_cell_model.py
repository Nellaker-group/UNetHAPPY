from pathlib import Path
from collections import namedtuple
from typing import List
import os

import typer
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    top_k_accuracy_score,
    cohen_kappa_score,
    matthews_corrcoef,
    precision_recall_curve,
    average_precision_score,
)

from happy.utils.utils import get_device
from happy.organs.organs import get_organ
from happy.train.cell_train import setup_data, setup_model


def main(
    project_name: str = typer.Option(...),
    organ_name: str = typer.Option(...),
    annot_dir: str = typer.Option(...),
    pre_trained: str = typer.Option(...),
    dataset_names: List[str] = typer.Option([]),
):
    """Evaluates model performance across validation datasets

    Args:
        project_name: name of the project dir to save visualisations to
        organ_name: name of organ for getting the cells
        annot_dir: relative path to annotations
        pre_trained: relative path to pretrained model
        dataset_names: the datasets who's validation set to evaluate over
    """
    device = get_device()

    project_dir = (
        Path(__file__).absolute().parent.parent.parent / "projects" / project_name
    )
    os.chdir(str(project_dir))

    HPs = namedtuple("HPs", "dataset_names batch")
    hp = HPs(dataset_names, 100)
    organ = get_organ(organ_name)
    cell_ids = [cell.id for cell in organ.cells]

    model, image_size = setup_model(
        "resnet-50", False, len(organ.cells), pre_trained, False, device
    )
    model.eval()

    multiple_val_sets = True if len(dataset_names) > 1 else False
    dataloaders = setup_data(
        organ,
        project_dir / annot_dir,
        hp,
        image_size,
        3,
        multiple_val_sets,
        hp.batch,
        False,
    )
    dataloaders.pop("train")

    print("Running inference across datasets")
    predictions = {}
    ground_truth = {}
    outs = {}
    with torch.no_grad():
        for dataset_name in dataloaders:
            predictions[dataset_name] = []
            ground_truth[dataset_name] = []
            outs[dataset_name] = []

            for data in tqdm(dataloaders[dataset_name]):
                ground_truths = data["annot"].tolist()

                out = model(data["img"].to(device).float())
                prediction = torch.max(out, 1)[1].cpu().tolist()
                out = out.cpu().detach().numpy()

                ground_truth[dataset_name].extend(ground_truths)
                predictions[dataset_name].extend(prediction)
                outs[dataset_name].extend(out)

    print("Evaluating datasets")
    for dataset_name in dataloaders:
        print(f"{dataset_name}:")
        accuracy = accuracy_score(ground_truth[dataset_name], predictions[dataset_name])
        cohen_kappa = cohen_kappa_score(
            ground_truth[dataset_name], predictions[dataset_name]
        )
        mcc = matthews_corrcoef(ground_truth[dataset_name], predictions[dataset_name])
        top_2_accuracy = top_k_accuracy_score(
            ground_truth[dataset_name], outs[dataset_name], k=2, labels=cell_ids
        )
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Top 2 accuracy: {top_2_accuracy:.3f}")
        print(f"Cohen's Kappa: {cohen_kappa:.3f}")
        print(f"MCC: {mcc:.3f}")

        alt_ground_truth = _convert_to_alt_label(organ, ground_truth[dataset_name])
        alt_predictions = _convert_to_alt_label(organ, predictions[dataset_name])
        alt_label_accuracy = accuracy_score(alt_ground_truth, alt_predictions)
        print(f"Alt Cell Label Accuracy: {alt_label_accuracy:.3f}")

        _plot_pr_curves(
            organ, dataset_name, ground_truth[dataset_name], outs[dataset_name]
        )


def _convert_to_alt_label(organ, labels):
    ids = [cell.id for cell in organ.cells]
    alt_ids = [cell.alt_id for cell in organ.cells]
    alt_id_mapping = dict(zip(ids, alt_ids))
    return [alt_id_mapping[label] for label in labels]


def _plot_pr_curves(organ, dataset_name, ground_truth, scores):
    cell_mapping = {cell.id: cell.label for cell in organ.cells}
    cell_classes = np.unique(list(cell_mapping.keys()))

    ground_truth = label_binarize(ground_truth, classes=cell_classes)
    scores = np.array(scores)
    scores = softmax(scores, axis=1)

    # Compute Precision-Recall and plot curve
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in cell_classes:
        precision[i], recall[i], _ = precision_recall_curve(
            ground_truth[:, i], scores[:, i]
        )
        average_precision[i] = average_precision_score(ground_truth[:, i], scores[:, i])

    # Compute micro-average ROC curve and ROC area
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        ground_truth.ravel(), scores.ravel()
    )
    average_precision["micro"] = average_precision_score(
        ground_truth, scores, average="micro"
    )

    # Plot Precision-Recall curve for each class
    plt.clf()
    plt.plot(
        recall["micro"],
        precision["micro"],
        label=f"mavg ({average_precision['micro']:0.2f})",
    )
    for i in cell_classes:
        plt.plot(
            recall[i],
            precision[i],
            label=f"{cell_mapping[i]} ({average_precision[i]:0.2f})",
        )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall curves for {dataset_name} cells")
    plt.legend(loc="lower right")
    plt.savefig(f"../../analysis/evaluation/plots/{dataset_name}_pr_curves.png")
    plt.clf()

if __name__ == "__main__":
    typer.run(main)
