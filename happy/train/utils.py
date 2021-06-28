import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import confusion_matrix


def get_confusion_matrix(organ, pred, truth):
    cell_labels = [cell.label for cell in organ.cells]
    cm = confusion_matrix(pred, truth)
    print(cm)
    return pd.DataFrame(cm, columns=cell_labels, index=cell_labels).astype(int)


def plot_confusion_matrix(cm, dataset_name, run_path):
    save_path = run_path / f"{dataset_name}_confusion_matrix.png"

    sns.heatmap(cm, annot=True, cmap="Blues", square=True, fmt="d")
    plt.title(f"Cell Classification for {dataset_name} Validation")
    plt.ylabel("True Label")
    plt.xlabel("Predicated Label")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    plt.clf()
