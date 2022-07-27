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
    run_id: int = typer.Option(...),
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
    cell_colours_mapping = {cell.label: cell.colourblind_colour for cell in organ.cells}
    project_dir = get_project_dir(project_name)

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
    unique_cell_counts = dict(zip(unique_cell_labels, cell_counts))
    print(f"Num cell predictions per label: {unique_cell_counts}")
    cell_proportions = [
        round((count / sum(cell_counts)) * 100, 2) for count in cell_counts
    ]
    unique_cell_proportions = dict(zip(unique_cell_labels, cell_proportions))
    print(f"Cell proportions per label: {unique_cell_proportions}")

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
    # remove rows where knots were removed
    if group_knts and not trained_with_grouped_knts:
        tissue_df = tissue_df.loc[~tissue_df.index.isin(inds_to_remove)].reset_index(
            drop=True
        )
    unique_tissues, tissue_counts = np.unique(tissue_df["class"], return_counts=True)
    unique_tissue_counts = dict(zip(unique_tissues, tissue_counts))
    print(f"Num tissue predictions per label: {unique_tissue_counts}")
    tissue_proportions = [
        round((count / sum(tissue_counts)) * 100, 2) for count in tissue_counts
    ]
    unique_tissue_proportions = dict(zip(unique_tissues, tissue_proportions))
    print(f"Tissue proportions per label: {unique_tissue_proportions}")

    # get number of cell types within each tissue type
    cell_df = pd.DataFrame(
        {"x": cell_coords[:, 0], "y": cell_coords[:, 1], "cells": predictions}
    )
    cell_df["cells"] = cell_df.cells.map(cell_label_mapping)
    cell_df.sort_values(by=["x", "y"], inplace=True, ignore_index=True)
    tissue_df.sort_values(by=["x", "y"], inplace=True, ignore_index=True)

    get_cells_within_tissues(
        organ,
        cell_df,
        tissue_df,
        cell_colours_mapping,
        pretrained_path,
        villus_only=True,
    )


# find cell types within each tissue type and plot as stacked bar chart
def get_cells_within_tissues(
    organ,
    cell_predictions,
    tissue_predictions,
    cell_colours_mapping,
    save_path,
    villus_only,
):
    tissue_label_to_name = {tissue.label: tissue.name for tissue in organ.tissues}
    cell_label_to_name = {cell.label: cell.name for cell in organ.cells}

    combined_df = pd.DataFrame(
        {
            "cell_x": cell_predictions["x"],
            "cell_y": cell_predictions["y"],
            "Cells": cell_predictions["cells"],
            "tissue_x": tissue_predictions["x"],
            "tissue_y": tissue_predictions["y"],
            "Tissues": tissue_predictions["class"],
        }
    )
    grouped_df = (
        combined_df.groupby(["Tissues", "Cells"]).size().reset_index(name="count")
    )
    grouped_df["prop"] = grouped_df.groupby(["Tissues"])["count"].transform(
        lambda x: x * 100 / x.sum()
    )
    prop_df = grouped_df.pivot_table(index="Tissues", columns="Cells", values="prop")
    prop_df = prop_df[reversed(prop_df.columns)]

    if villus_only:
        prop_df = prop_df.drop(["Fibrin", "Avascular", "Maternal", "AVilli"], axis=0)
        prop_df = prop_df.reindex(["Chorion", "SVilli", "MIVilli", "TVilli", "Sprout"])

    prop_df.index = prop_df.index.map(tissue_label_to_name)
    cell_colours = [cell_colours_mapping[cell] for cell in prop_df.columns]
    prop_df.columns = prop_df.columns.map(cell_label_to_name)

    sns.set(style="whitegrid")
    plt.figure(figsize=(16, 16))
    ax = prop_df.plot(
        kind="bar", stacked=True, color=cell_colours, legend="reverse", width=0.7
    )
    ax.set(xlabel=None)
    ticks_and_labels = plt.xticks(
        range(len(prop_df.index)), list(prop_df.index), rotation=0, size=9
    )
    for i, label in enumerate(ticks_and_labels[1]):
        label.set_y(label.get_position()[1] - (i % 2) * 0.05)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[::-1],
        labels[::-1],
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        prop={"size": 10},
    )
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height])
    plt.tight_layout()
    plt.savefig(save_path / "cells_in_tissues.png")
    print(f"Plot saved to {save_path / 'cells_in_tissues.png'}")


if __name__ == "__main__":
    typer.run(main)
