from pathlib import Path

import typer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from happy.train.utils import get_tissue_confusion_matrix, plot_confusion_matrix
from happy.organs.organs import get_organ
from happy.utils.utils import get_project_dir
from projects.placenta.graphs.graphs.create_graph import get_groundtruth_patch


def main(
    project_name: str = "placenta",
    exp_name: str = typer.Option(...),
    model_weights_dir: str = typer.Option(...),
    model_name: str = typer.Option(...),
    model_type: str = "sup_clustergcn",
    remove_unlabelled: bool = True,
):
    organ = get_organ("placenta")
    label_id_mapping = {tissue.label: tissue.id for tissue in organ.tissues}
    project_dir = get_project_dir(project_name)

    # Get ground truth manually annotated data
    _, _, tissue_class = get_groundtruth_patch(
        organ, project_dir, 0, 0, -1, -1, "96_tissue_points.tsv", "full"
    )

    # print tissue predictions from tsv file
    pretrained_path = (
        project_dir
        / "results"
        / "graph"
        / model_type
        / exp_name
        / model_weights_dir
        / "eval"
        / model_name
    )
    tissue_df = pd.read_csv(pretrained_path / "tissue_preds.tsv", sep="\t")

    predictions = [label_id_mapping[tissue] for tissue in np.array(tissue_df['class'])]
    cm_df, cm_df_props = get_tissue_confusion_matrix(
        organ, predictions, tissue_class, remove_unlabelled, proportion_label=True
    )
    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1.1)
    plot_confusion_matrix(cm_df, "All Tissues", Path("plots"), "d")
    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1.1)
    plot_confusion_matrix(cm_df_props, "All Tissues Proportion", Path("plots"), ".2f")


if __name__ == "__main__":
    typer.run(main)
