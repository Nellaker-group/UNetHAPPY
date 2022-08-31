from typing import List

import typer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from happy.organs.organs import get_organ
from happy.utils.utils import get_project_dir
import happy.db.eval_runs_interface as db
from happy.hdf5.utils import get_datasets_in_patch, get_embeddings_file
from projects.placenta.graphs.analysis.knot_nuclei_to_point import process_knt_cells


def main(
    run_ids: List[int] = typer.Option([]),
    project_name: str = "placenta",
    exp_name: str = typer.Option(...),
    model_weights_dir: str = typer.Option(...),
    model_name: str = typer.Option(...),
    model_type: str = "sup_clustergcn",
    x_min: int = 0,
    y_min: int = 0,
    width: int = -1,
    height: int = -1,
    group_knts: bool = False,
    trained_with_grouped_knts: bool = False,
):
    # Create database connection
    db.init()
    organ = get_organ("placenta")
    cell_label_mapping = {cell.id: cell.label for cell in organ.cells}
    project_dir = get_project_dir(project_name)

    cell_prop_dfs = []
    tissue_prop_dfs = []
    for run_id in run_ids:
        # Get path to embeddings hdf5 files
        embeddings_path = get_embeddings_file(project_name, run_id)
        # Get hdf5 datasets contained in specified box/patch of WSI
        predictions, embeddings, cell_coords, confidence = get_datasets_in_patch(
            embeddings_path, x_min, y_min, width, height
        )

        if group_knts:
            (
                predictions,
                embeddings,
                cell_coords,
                confidence,
                inds_to_remove,
            ) = process_knt_cells(
                predictions, embeddings, cell_coords, confidence, organ, 50, 3
            )

        # print cell predictions from hdf5 file
        unique_cells, cell_counts = np.unique(predictions, return_counts=True)
        unique_cell_labels = []
        for label in unique_cells:
            unique_cell_labels.append(cell_label_mapping[label])
        cell_proportions = [
            round((count / sum(cell_counts)), 2) for count in cell_counts
        ]
        unique_cell_proportions = dict(zip(unique_cell_labels, cell_proportions))
        cell_prop_dfs.append(pd.DataFrame([unique_cell_proportions]))

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
            / f"run_{run_id}"
        )
        tissue_df = pd.read_csv(pretrained_path / "tissue_preds.tsv", sep="\t")
        # remove rows where knots were removed
        if group_knts and not trained_with_grouped_knts:
            tissue_df = tissue_df.loc[
                ~tissue_df.index.isin(inds_to_remove)
            ].reset_index(drop=True)
        unique_tissues, tissue_counts = np.unique(
            tissue_df["class"], return_counts=True
        )
        tissue_proportions = [
            round((count / sum(tissue_counts)), 2) for count in tissue_counts
        ]
        unique_tissue_proportions = dict(zip(unique_tissues, tissue_proportions))
        tissue_prop_dfs.append(pd.DataFrame([unique_tissue_proportions]))

    cell_df = pd.concat(cell_prop_dfs)
    args_to_sort = np.argsort([cell.structural_id for cell in organ.cells])
    cell_df = cell_df[cell_df.columns[args_to_sort]]
    cell_colours = {cell.label: cell.colourblind_colour for cell in organ.cells}

    tissue_df = pd.concat(tissue_prop_dfs)
    tissue_labels = [tissue.label for tissue in organ.tissues]
    args_to_sort = [
        np.where(tissue_df.columns.to_numpy() == np.array(tissue_labels)[:, None])[1]
    ]
    tissue_df = tissue_df[tissue_df.columns[args_to_sort]]
    tissue_colours = {
        tissue.label: tissue.colourblind_colour for tissue in organ.tissues
    }
    print(cell_df)
    print(tissue_df)

    plot_box_and_whisker(cell_df, "plots/cell_proportions.png", "Cell", cell_colours)

    TISSUE_EXPECTATION = [
        (0.0, 0.009),
        (0.30, 0.60),
        (0.17, 0.32),
        (None, None),
        (0.09, 0.25),
        (None, None),
        (None, None),
        (0.0, 0.09),
        (0.0, 0.0249),
    ]
    plot_box_and_whisker(
        tissue_df,
        "plots/tissue_proportions.png",
        "Tissue",
        tissue_colours,
        expectation=TISSUE_EXPECTATION,
    )


def plot_box_and_whisker(df, save_path, entity, colours, expectation=None):
    sns.set(style="whitegrid")
    plt.subplots(figsize=(10, 8), dpi=400)
    ax = sns.boxplot(data=df, palette=colours, whis=[0, 100])
    sns.swarmplot(data=df, color=".25")
    plt.ylim(top=0.62)
    ax.set(ylabel=f"Proportion of {entity}s Across WSIs")
    ax.set(xlabel=f"{entity} Labels")
    plt.tight_layout()
    if expectation is not None:
        for i, (low, high) in enumerate(expectation):
            if low is not None:
                ax.hlines(
                    y=low,
                    xmin=i - 0.5,
                    xmax=i + 0.5,
                    color="lime",
                    linestyles="dashed",
                    linewidth=3,
                )
            if high is not None:
                ax.hlines(
                    y=high,
                    xmin=i - 0.5,
                    xmax=i + 0.5,
                    color="lime",
                    linestyles="dashed",
                    linewidth=3,
                )
    plt.savefig(save_path)
    plt.close()
    plt.clf()


if __name__ == "__main__":
    typer.run(main)
