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
from projects.placenta.graphs.graphs.create_graph import get_groundtruth_patch
from projects.placenta.graphs.graphs.create_graph import get_nodes_within_tiles


def main(
    run_id: int = typer.Option(...),
    project_name: str = "placenta",
    tissue_label_tsv: str = typer.Option(...),
    patch_files: List[str] = None,
    group_knts: bool = False,
    use_path_class: bool = False,
):
    # Create database connection
    db.init()
    organ = get_organ("placenta")
    cell_label_mapping = {cell.id: cell.label for cell in organ.cells}
    cell_colours_mapping = {cell.label: cell.colourblind_colour for cell in organ.cells}
    tissue_label_mapping = {tissue.id: tissue.label for tissue in organ.tissues}
    project_dir = get_project_dir(project_name)
    patch_files = [project_dir / "config" / file for file in patch_files]

    # Get path to embeddings hdf5 files
    embeddings_path = get_embeddings_file(project_name, run_id)
    # Get hdf5 datasets contained in specified box/patch of WSI
    predictions, embeddings, cell_coords, confidence = get_datasets_in_patch(
        embeddings_path, 0, 0, -1, -1
    )
    # Get ground truth manually annotated data
    xs_tissue, ys_tissue, tissue_class = get_groundtruth_patch(
        organ, project_dir, 0, 0, -1, -1, tissue_label_tsv
    )
    sort_args = np.lexsort((cell_coords[:, 1], cell_coords[:, 0]))
    predictions = predictions[sort_args]
    embeddings = embeddings[sort_args]
    cell_coords = cell_coords[sort_args]
    confidence = confidence[sort_args]

    # TODO: get the tissue pathologist 'ground truth' from patches
    # Get patch of interest from patch file
    if len(patch_files) != 0:
        patch_node_inds = []
        path_class = []
        for file in patch_files:
            patches_df = pd.read_csv(file)
            for row in patches_df.itertuples(index=False):
                patch_node_inds.extend(
                    get_nodes_within_tiles(
                        (row.x, row.y), row.width, row.height, xs_tissue, ys_tissue
                    )
                )
                if use_path_class:
                    path_class.append(row.path_class)
        # sort_args = np.lexsort((ys_tissue, xs_tissue))

    # Otherwise, use all tissue nodes and remove unlabelled
    else:
        patch_node_inds = tissue_class.nonzero()[0]
        print(f"Removing {len(tissue_class) - len(patch_node_inds)} unlabelled nodes")

    predictions = predictions[patch_node_inds]
    embeddings = embeddings[patch_node_inds]
    cell_coords = cell_coords[patch_node_inds]
    confidence = confidence[patch_node_inds]
    xs_tissue = xs_tissue[patch_node_inds]
    ys_tissue = ys_tissue[patch_node_inds]
    tissue_class = tissue_class[patch_node_inds]

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

    # # print tissue ground truth
    tissue_df = pd.DataFrame({"x": xs_tissue, "y": ys_tissue, "Tissues": tissue_class})
    tissue_df['Tissues'] = tissue_df['Tissues'].map(tissue_label_mapping)
    # remove rows where knots were removed
    if group_knts:
        tissue_df = tissue_df.loc[~tissue_df.index.isin(inds_to_remove)].reset_index(
            drop=True
        )
    unique_tissues, tissue_counts = np.unique(tissue_df["Tissues"], return_counts=True)
    unique_tissue_counts = dict(zip(unique_tissues, tissue_counts))
    print(f"Num tissue predictions per label: {unique_tissue_counts}")
    tissue_proportions = [
        round((count / sum(tissue_counts)) * 100, 2) for count in tissue_counts
    ]
    unique_tissue_proportions = dict(zip(unique_tissues, tissue_proportions))
    print(f"Tissue proportions per label: {unique_tissue_proportions}")

    # get number of cell types within each tissue type
    cell_df = pd.DataFrame(
        {"x": cell_coords[:, 0], "y": cell_coords[:, 1], "Cells": predictions}
    )
    cell_df["Cells"] = cell_df.Cells.map(cell_label_mapping)
    cell_df.sort_values(by=["x", "y"], inplace=True, ignore_index=True)
    tissue_df.sort_values(by=["x", "y"], inplace=True, ignore_index=True)

    get_cells_within_tissues(
        organ,
        cell_df,
        tissue_df,
        cell_colours_mapping,
        project_dir / "analysis" / "plots",
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

    combined_df = pd.merge(cell_predictions, tissue_predictions)

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

    sns.set(style="white")
    plt.rcParams["figure.dpi"] = 600
    ax = prop_df.plot(
        kind="bar",
        stacked=True,
        color=cell_colours,
        legend="reverse",
        width=0.8,
        figsize=(8.5, 6),
    )
    ax.set(xlabel=None)
    plt.xticks(range(len(prop_df.index)), list(prop_df.index), rotation=0, size=9)
    plt.yticks([])
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[::-1],
        labels[::-1],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        prop={"size": 9.25},
        ncol=4,
    )
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height])
    plt.tight_layout()
    sns.despine(left=True)
    plt.savefig(save_path / "cells_in_tissues.png")
    print(f"Plot saved to {save_path / 'cells_in_tissues.png'}")


if __name__ == "__main__":
    typer.run(main)
