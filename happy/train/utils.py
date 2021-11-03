from datetime import datetime

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix


def get_confusion_matrix(organ, pred, truth):
    cell_labels = [cell.label for cell in organ.cells]
    cm = confusion_matrix(pred, truth)
    return pd.DataFrame(cm, columns=cell_labels, index=cell_labels).astype(int)


def plot_confusion_matrix(cm, dataset_name, run_path, fmt="d"):
    save_path = run_path / f"{dataset_name}_confusion_matrix.png"

    sns.heatmap(cm, annot=True, cmap="Blues", square=True, fmt=fmt)
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
