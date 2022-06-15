from datetime import datetime

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix


def get_cell_confusion_matrix(organ, pred, truth):
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
            cm = np.hstack((cm[:,:missing_id], column_insert, cm[:,missing_id:]))
            row_insert = np.zeros((1, cm.shape[1]))
            cm = np.insert(cm, missing_id, row_insert, 0)

    return pd.DataFrame(cm, columns=cell_labels, index=cell_labels).astype(int)


def plot_confusion_matrix(cm, dataset_name, run_path, fmt="d"):
    save_path = run_path / f"{dataset_name}_confusion_matrix.png"

    sns.heatmap(cm, annot=True, cmap="Blues", square=True, cbar=False, fmt=fmt)
    plt.title(f"Classification for {dataset_name} Validation")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    plt.clf()


def setup_run(project_dir, exp_name, dataset_type):
    fmt = "%Y-%m-%dT%H-%M-%S"
    timestamp = datetime.strftime(datetime.utcnow(), fmt)
    run_path = project_dir / "results" / dataset_type / exp_name / timestamp
    run_path.mkdir(parents=True, exist_ok=True)
    return run_path
