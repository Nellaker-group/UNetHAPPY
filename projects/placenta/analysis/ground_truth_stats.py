from typing import List

import typer
import numpy as np
import pandas as pd

from happy.organs import get_organ
from happy.utils.utils import get_project_dir
import happy.db.eval_runs_interface as db
from happy.graph.graph_creation.get_and_process import get_raw_data
from projects.placenta.graphs.processing.process_knots import process_knts
from happy.graph.graph_creation.create_graph import get_nodes_within_tiles
from projects.placenta.analysis.wsi_cell_tissue_stats import get_cells_within_tissues


def main(
    run_ids: List[int] = typer.Option(...),
    project_name: str = "placenta",
    tissue_label_tsvs: List[str] = typer.Option(...),
    patch_files: List[str] = None,
    group_knts: bool = False,
    use_path_class: bool = False,
):
    # Create database connection
    db.init()
    organ = get_organ("placenta")
    cell_label_mapping = {cell.id: cell.label for cell in organ.cells}
    cell_colours_mapping = {cell.label: cell.colour for cell in organ.cells}
    tissue_label_mapping = {tissue.id: tissue.label for tissue in organ.tissues}
    project_dir = get_project_dir(project_name)
    patch_files = [project_dir / "graph_splits" / file for file in patch_files]

    cell_dfs = []
    tissue_dfs = []
    for i, run_id in enumerate(run_ids):
        # Get raw data and ground truth data
        hdf5_data, tissue_class = get_raw_data(
            project_name, organ, project_dir, run_id, 0, 0, -1, -1, tissue_label_tsvs[i]
        )

        # Get patch of interest from patch file
        if len(patch_files) != 0:
            patch_node_inds = []
            path_class = []
            for file in patch_files:
                patches_df = pd.read_csv(file)
                for row in patches_df.itertuples(index=False):
                    node_inds = get_nodes_within_tiles(
                        (row.x, row.y),
                        row.width,
                        row.height,
                        hdf5_data.coords[:, 0],
                        hdf5_data.coords[:, 1],
                    )
                    patch_node_inds.extend(node_inds)
                    if use_path_class:
                        path_class.extend([row.path_class] * len(node_inds))
        # Otherwise, use all tissue nodes and remove unlabelled
        else:
            patch_node_inds = tissue_class.nonzero()[0]
            print(
                f"Removing {len(tissue_class) - len(patch_node_inds)} unlabelled nodes"
            )

        hdf5_data = hdf5_data._apply_mask(patch_node_inds)
        tissue_class = tissue_class[patch_node_inds]

        if group_knts:
            hdf5_data, tissue_class = process_knts(organ, hdf5_data, tissue_class)

        # print cell predictions from hdf5 file
        unique_cells, cell_counts = np.unique(
            hdf5_data.cell_predictions, return_counts=True
        )
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
        if not use_path_class:
            tissue_df = pd.DataFrame(
                {
                    "x": hdf5_data.coords[:, 0],
                    "y": hdf5_data.coords[:, 1],
                    "Tissues": tissue_class,
                }
            )
            tissue_df["Tissues"] = tissue_df["Tissues"].map(tissue_label_mapping)
        else:
            tissue_df = pd.DataFrame(
                {
                    "x": hdf5_data.coords[:, 0],
                    "y": hdf5_data.coords[:, 1],
                    "Tissues": path_class,
                }
            )
        unique_tissues, tissue_counts = np.unique(
            tissue_df["Tissues"], return_counts=True
        )
        unique_tissue_counts = dict(zip(unique_tissues, tissue_counts))
        print(f"Num tissue predictions per label: {unique_tissue_counts}")
        tissue_proportions = [
            round((count / sum(tissue_counts)) * 100, 2) for count in tissue_counts
        ]
        unique_tissue_proportions = dict(zip(unique_tissues, tissue_proportions))
        print(f"Tissue proportions per label: {unique_tissue_proportions}")

        # get number of cell types within each tissue type
        cell_df = pd.DataFrame(
            {
                "x": hdf5_data.coords[:, 0],
                "y": hdf5_data.coords[:, 1],
                "Cells": hdf5_data.cell_predictions,
            }
        )
        cell_df["Cells"] = cell_df.Cells.map(cell_label_mapping)
        cell_df.sort_values(by=["x", "y"], inplace=True, ignore_index=True)
        tissue_df.sort_values(by=["x", "y"], inplace=True, ignore_index=True)
        cell_dfs.append(cell_df)
        tissue_dfs.append(tissue_df)
    full_cell_df = pd.concat(cell_dfs)
    full_tissue_df = pd.concat(tissue_dfs)

    get_cells_within_tissues(
        organ,
        full_cell_df,
        full_tissue_df,
        cell_colours_mapping,
        project_dir / "analysis" / "plots" / "cells_in_tissues_gt.png",
        villus_only=False,
    )


if __name__ == "__main__":
    typer.run(main)
