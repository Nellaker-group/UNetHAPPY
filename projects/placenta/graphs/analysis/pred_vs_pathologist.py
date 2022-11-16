from typing import List
from pathlib import Path

import typer
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    cohen_kappa_score,
    matthews_corrcoef,
)
import matplotlib.pyplot as plt
import seaborn as sns

from happy.organs import get_organ
from happy.utils.utils import get_project_dir
import happy.db.eval_runs_interface as db
from projects.placenta.graphs.graphs.create_graph import get_nodes_within_tiles
from happy.train.utils import plot_confusion_matrix, get_tissue_confusion_matrix


def main(
    run_id: int = typer.Option(...),
    project_name: str = "placenta",
    exp_name: str = typer.Option(...),
    model_weights_dir: str = typer.Option(...),
    model_name: str = typer.Option(...),
    model_type: str = "sup_clustergcn",
    patch_files: List[str] = typer.Option(...),
):
    # Create database connection
    db.init()
    organ = get_organ("placenta")
    project_dir = get_project_dir(project_name)
    patch_files = [project_dir / "graph_splits" / file for file in patch_files]

    # print tissue predictions from tsv file
    pretrained_path = (
        project_dir
        / "results"
        / "graph"
        / model_type
        / exp_name
        / model_weights_dir
        / "cell_infer"
        / model_name
        / f"run_{run_id}"
    )
    tissue_df = pd.read_csv(pretrained_path / "tissue_preds.tsv", sep="\t")

    # Get patch of interest from patch file
    patch_node_inds = []
    path_class = []
    for file in patch_files:
        patches_df = pd.read_csv(file)
        for row in patches_df.itertuples(index=False):
            node_inds = get_nodes_within_tiles(
                (row.x, row.y),
                row.width,
                row.height,
                tissue_df["x"].to_numpy(),
                tissue_df["y"].to_numpy(),
            )
            patch_node_inds.extend(node_inds)
            path_class.extend([row.path_class] * len(node_inds))

    tissue_df = tissue_df.filter(items=patch_node_inds, axis=0)

    # Setup paths
    save_path = pretrained_path / "pathologist_comparison"
    save_path.mkdir(parents=True, exist_ok=True)

    # Evaluate
    accuracy = accuracy_score(tissue_df["class"], path_class)
    balanced_accuracy = balanced_accuracy_score(tissue_df["class"], path_class)
    f1_macro = f1_score(tissue_df["class"], path_class, average="macro")
    cohen_kappa = cohen_kappa_score(tissue_df["class"], path_class)
    mcc = matthews_corrcoef(tissue_df["class"], path_class)
    print("-----------------------")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Balanced accuracy: {balanced_accuracy:.3f}")
    print(f"F1 macro score: {f1_macro:.3f}")
    print(f"Cohen's Kappa score: {cohen_kappa:.3f}")
    print(f"MCC score: {mcc:.3f}")
    print("-----------------------")

    tissue_label_to_ids = {tissue.label: tissue.id for tissue in organ.tissues}
    tissue_df['class'] = tissue_df['class'].map(tissue_label_to_ids)
    path_class = [tissue_label_to_ids[label] for label in path_class]

    print("Plotting confusion matrices")
    cm_df, cm_df_props = get_tissue_confusion_matrix(
        organ, tissue_df["class"], path_class, proportion_label=True
    )
    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1.1)
    plot_confusion_matrix(cm_df, "All Tissues", save_path, "d")
    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1.1)
    plot_confusion_matrix(cm_df_props, "All Tissues Proportion", save_path, ".2f")


if __name__ == "__main__":
    typer.run(main)
