from pathlib import Path

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def confusion_matrix(pred, truth):
    flatten = lambda l: [item for sublist in l for item in sublist]
    predicted_cell = flatten(pred)
    truth_cell = flatten(truth)
    cm = confusion_matrix(predicted_cell, truth_cell)
    print(cm)
    return pd.DataFrame(
        cm,
        columns=["CYT", "FIB", "HOF", "SYN", "VEN"],
        index=["CYT", "FIB", "HOF", "SYN", "VEN"],
    ).astype(int)


def plot_confusion_matrix(cm, dataset_name, run_path):
    run_path = Path(run_path)
    save_path = run_path / f"{dataset_name}_confusion_matrix.png"
    save_path.mkdir(parents=True, exist_ok=True)

    sns.heatmap(cm, annot=True, cmap="Blues", square=True, fmt="d")
    plt.title(f"Cell Classification for {dataset_name} Validation")
    plt.ylabel("True Label")
    plt.xlabel("Predicated Label")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    plt.clf()
