from datetime import datetime

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import label_binarize
from scipy.special import softmax
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
)


def setup_run(project_dir, exp_name, dataset_type):
    fmt = "%Y-%m-%dT%H-%M-%S"
    timestamp = datetime.strftime(datetime.utcnow(), fmt)
    run_path = project_dir / "results" / dataset_type / exp_name / timestamp
    run_path.mkdir(parents=True, exist_ok=True)
    return run_path


def get_cell_confusion_matrix(organ, pred, truth, proportion_label=False):
    cell_labels = [cell.label for cell in organ.cells]
    cell_ids = {cell.id for cell in organ.cells}

    unique_values_in_pred = set(pred)
    unique_values_in_truth = set(truth)
    unique_values_in_matrix = unique_values_in_pred.union(unique_values_in_truth)
    missing_cell_ids = list(cell_ids - unique_values_in_matrix)
    missing_cell_ids.sort()

    cm = confusion_matrix(truth, pred)

    if len(missing_cell_ids) > 0:
        for missing_id in missing_cell_ids:
            column_insert = np.zeros((cm.shape[0], 1))
            cm = np.hstack((cm[:, :missing_id], column_insert, cm[:, missing_id:]))
            row_insert = np.zeros((1, cm.shape[1]))
            cm = np.insert(cm, missing_id, row_insert, 0)

    row_labels = []
    if proportion_label:
        unique_counts = cm.sum(axis=1)
        total_counts = cm.sum()
        label_proportions = ((unique_counts / total_counts) * 100).round(2)
        for i, label in enumerate(cell_labels):
            row_labels.append(f"{label}\n({label_proportions[i]}%)")

    return pd.DataFrame(cm, columns=cell_labels, index=row_labels).astype(int)


def get_tissue_confusion_matrix(
    organ, pred, truth, remove_unlabelled=True, proportion_label=False
):
    tissue_ids = {tissue.id for tissue in organ.tissues}
    tissue_labels = {tissue.label for tissue in organ.tissues}

    unique_values_in_pred = set(pred)
    unique_values_in_truth = set(truth)
    unique_values_in_matrix = unique_values_in_pred.union(unique_values_in_truth)
    missing_tissue_ids = list(set(tissue_ids) - unique_values_in_matrix)
    missing_tissue_ids.sort()

    if remove_unlabelled:
        if 0 in missing_tissue_ids:
            missing_tissue_ids.remove(0)
            tissue_labels = tissue_labels[1:]

    cm = confusion_matrix(truth, pred)

    if len(missing_tissue_ids) > 0:
        for missing_id in missing_tissue_ids:
            column_insert = np.zeros((cm.shape[0], 1))
            cm = np.hstack((cm[:, :missing_id], column_insert, cm[:, missing_id:]))
            row_insert = np.zeros((1, cm.shape[1]))
            cm = np.insert(cm, missing_id, row_insert, 0)

    row_labels = []
    if proportion_label:
        unique_counts = cm.sum(axis=1)
        total_counts = cm.sum()
        label_proportions = ((unique_counts / total_counts) * 100).round(2)
        for i, label in enumerate(tissue_labels):
            row_labels.append(f"{label} ({label_proportions[i]}%)")

    cm_df = pd.DataFrame(cm, columns=tissue_labels, index=tissue_labels).astype(int)
    unique_counts = cm.sum(axis=1)

    cm_df_props = (
        pd.DataFrame(
            cm / unique_counts[:, None], columns=tissue_labels, index=tissue_labels
        )
        .fillna(0)
        .astype(float)
    )

    empty_rows = (cm_df.T != 0).any()
    cm_df = cm_df[empty_rows]
    cm_df_props = cm_df_props[empty_rows]
    empty_row_names = empty_rows[empty_rows == False].index.tolist()
    cm_df = cm_df.drop(columns=empty_row_names)
    cm_df_props = cm_df_props.drop(columns=empty_row_names)

    row_labels = np.array(row_labels)
    row_labels = row_labels[empty_rows]
    cm_df_props.set_index(row_labels, drop=True, inplace=True)

    return cm_df, cm_df_props


def plot_confusion_matrix(cm, dataset_name, run_path, fmt="d"):
    save_path = run_path / f"{dataset_name}_confusion_matrix.png"

    sns.heatmap(cm, annot=True, cmap="Blues", square=True, cbar=False, fmt=fmt)
    # plt.title(f"Classification for {dataset_name} Validation")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    plt.clf()


def plot_pr_curves(id_to_label, colours, ground_truth, scores, save_path, figsize=None):
    class_ids = np.unique(list(id_to_label.keys()))

    ground_truth = label_binarize(ground_truth, classes=class_ids)
    scores = np.array(scores)
    scores = softmax(scores, axis=-1)

    # Compute Precision-Recall and plot curve
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in class_ids:
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
    plt.figure(figsize=figsize)
    ax = plt.subplot(111)
    plt.plot(
        recall["micro"],
        precision["micro"],
        label=f"mavg ({average_precision['micro']:0.2f})",
        color="black",
    )
    for i in class_ids:
        plt.plot(
            recall[i],
            precision[i],
            label=f"{id_to_label[i]} ({average_precision[i]:0.2f})",
            color=colours[i],
        )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    plt.savefig(save_path)
    plt.clf()
