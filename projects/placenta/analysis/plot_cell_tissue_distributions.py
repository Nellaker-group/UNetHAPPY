from typing import List, Optional

import typer
import numpy as np
import pandas as pd
import matplotlib, matplotlib.pyplot as plt
import seaborn as sns

from happy.organs import get_organ
from happy.utils.utils import get_project_dir
import happy.db.eval_runs_interface as db
from happy.hdf5 import get_datasets_in_patch, get_embeddings_file
from projects.placenta.graphs.processing.process_knots import process_knt_cells


def main(
    run_ids: Optional[List[int]] = typer.Option([]),
    file_run_ids: Optional[str] = None,
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
    include_counts: bool = False,
    include_tissues: bool = True,
):
    """Plot the distribution of cell and tissue types across multiple WSIs.
    This will plot a line plot with an offset swarm plot where each point is each WSI.
    """

    # Create database connection
    db.init()
    organ = get_organ("placenta")
    cell_label_mapping = {cell.id: cell.name for cell in organ.cells}
    project_dir = get_project_dir(project_name)

    if file_run_ids is not None:
        run_ids = (
            pd.read_csv(project_dir / file_run_ids, header=None)
            .values.flatten()
            .tolist()
        )

    cell_prop_dfs = []
    cell_counts_df = []
    tissue_prop_dfs = []
    tissue_counts_df = []
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

        if include_counts:
            # calculate the area using the tiles containing nuclei to get cells per area
            tile_coords = np.array(db.get_run_state(run_id))
            tile_width = tile_coords[tile_coords[:, 1].argmax() + 1][0]
            tile_height = tile_coords[1][1]
            xs = cell_coords[:, 0]
            ys = cell_coords[:, 1]

            # count how many tiles have at least one cell
            tile_count = 0
            for tile in tile_coords:
                tile_x = tile[0]
                tile_y = tile[1]

                mask = np.logical_and(
                    (np.logical_and(xs >= tile_x, (ys >= tile_y))),
                    (
                        np.logical_and(
                            xs <= (tile_x + tile_width), (ys <= (tile_y + tile_height))
                        )
                    ),
                )
                if np.any(mask):
                    tile_count += 1
            print(f"Number of tiles with a least one cell: {tile_count}")
            tile_area = tile_count * tile_width * tile_height
            slide_pixel_size = db.get_slide_pixel_size_by_evalrun(run_id)
            tile_area = tile_area * slide_pixel_size * slide_pixel_size
            print(f"Tile area in um^2: {tile_area}")

            cell_counts = [count / tile_area * 1000000 for count in cell_counts]
            all_cell_counts = dict(zip(unique_cell_labels, cell_counts))
            cell_counts_df.append(pd.DataFrame([all_cell_counts]))

        if include_tissues:
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

            tissue_names_mapping = {
                tissue.label: tissue.name for tissue in organ.tissues
            }
            tissue_df["class"] = tissue_df["class"].map(tissue_names_mapping)
            unique_tissues, tissue_counts = np.unique(
                tissue_df["class"], return_counts=True
            )
            tissue_proportions = [
                round((count / sum(tissue_counts)), 2) for count in tissue_counts
            ]
            unique_tissue_proportions = dict(zip(unique_tissues, tissue_proportions))
            tissue_prop_dfs.append(pd.DataFrame([unique_tissue_proportions]))

            if include_counts and include_tissues:
                tissue_counts = [count / tile_area * 1000000 for count in tissue_counts]
                all_tissue_counts = dict(zip(unique_tissues, tissue_counts))
                tissue_counts_df.append(pd.DataFrame([all_tissue_counts]))

    cell_df = _reorder_cell_columns(pd.concat(cell_prop_dfs), organ)
    cell_colours = {cell.name: cell.colour for cell in organ.cells}
    cell_colours["Total"] = "#000000"
    plot_distribution(
        cell_df,
        "plots/cell_proportions.png",
        "Cell",
        cell_colours,
        ylabel="Proportion of Cells Across WSIs",
    )
    cell_df.to_csv("plots/cell_proportions.csv")

    if include_tissues:
        TISSUE_EXPECTATION = [
            (0.0, 0.009),
            (0.30, 0.60),
            (0.17, 0.32),
            (None, None),
            (0.09, 0.25),
            (None, None),
            (None, None),
            (0.0, 0.10),
            (0.0, 0.0249),
        ]

        tissue_df = _reorder_tissue_columns(pd.concat(tissue_prop_dfs), organ)
        tissue_colours = {tissue.name: tissue.colour for tissue in organ.tissues}
        plot_distribution(
            tissue_df,
            "plots/tissue_proportions.png",
            "Tissue",
            tissue_colours,
            ylabel="Proportion of Tissues Across WSIs",
        )
        tissue_df.to_csv("plots/tissue_proportions.csv")

    if include_counts:
        cell_counts_df = _reorder_cell_columns(pd.concat(cell_counts_df), organ)
        cell_counts_df["Total"] = cell_counts_df[list(cell_counts_df.columns)].sum(
            axis=1
        )
        plot_distribution(
            cell_counts_df,
            "plots/cell_counts.png",
            "Cell",
            cell_colours,
            ylim=6000.0,
            bottom=-200,
            ylabel="Number of Cells / mm^2",
        )
        cell_counts_df.to_csv("plots/cell_counts.csv")

        if include_tissues:
            tissue_counts_df = _reorder_tissue_columns(
                pd.concat(tissue_counts_df), organ
            )
            tissue_counts_df["Total"] = tissue_counts_df[
                list(tissue_counts_df.columns)
            ].sum(axis=1)
            tissue_colours["Total"] = "#000000"
            plot_distribution(
                tissue_counts_df,
                "plots/tissue_counts.png",
                "Tissue",
                tissue_colours,
                ylim=6000.0,
                bottom=-200,
                ylabel="Number of Nuclei in Tissues / mm^2",
            )
            tissue_counts_df.to_csv("plots/tissue_counts.csv")


def plot_distribution(
    df,
    save_path,
    entity,
    colours,
    box=False,
    swarm=False,
    violin=False,
    cat=False,
    expectation=None,
    ylim=0.72,
    bottom=-0.02,
    ylabel=None,
):
    sns.set_style("white")
    plt.subplots(figsize=(8, 8), dpi=400)
    if box:
        ax = sns.boxplot(data=df, palette=colours, fliersize=1.5)
        if swarm:
            ax = sns.swarmplot(data=df, color=".5", size=1.5)
    elif swarm:
        ax = sns.swarmplot(data=df, palette=colours, size=1)
    elif violin:
        ax = sns.violinplot(data=df, palette=colours, bw=1., cut=0)
    elif cat:
        ax = sns.catplot(data=df, palette=colours, kind='violin')
    else:
        ax = sns.swarmplot(data=df, color=".5", size=1)
        _offset_swarm(ax, 0.3)

        melted_df = pd.melt(df.reset_index(drop=True).T.reset_index(), id_vars="index")
        ax = sns.lineplot(
            data=melted_df,
            x="index",
            y="value",
            hue="index",
            marker="o",
            err_style="bars",
            palette=colours,
            markersize=15,
            legend=False,
        )
        ax.lines[0].set_linestyle("")

    plt.ylim(bottom=bottom, top=ylim)
    ax.set(ylabel=ylabel)
    ax.set(xlabel=f"{entity} Labels")
    plt.xticks(rotation=90)
    plt.tight_layout()
    if expectation is not None:
        _add_expectation(ax, expectation)
    plt.savefig(save_path)
    plt.close()
    plt.clf()


def _offset_swarm(ax, offset):
    path_collections = [
        child
        for child in ax.get_children()
        if isinstance(child, matplotlib.collections.PathCollection)
    ]
    for path_collection in path_collections:
        x, y = np.array(path_collection.get_offsets()).T
        xnew = x + offset
        offsets = list(zip(xnew, y))
        path_collection.set_offsets(offsets)


def _add_expectation(ax, expectation):
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


def _reorder_cell_columns(cell_df, organ):
    args_to_sort = np.argsort([cell.structural_id for cell in organ.cells])
    return cell_df[cell_df.columns[args_to_sort]]


def _reorder_tissue_columns(tissue_df, organ):
    tissue_labels = [tissue.name for tissue in organ.tissues]
    args_to_sort = [
        np.where(tissue_df.columns.to_numpy() == np.array(tissue_labels)[:, None])[1]
    ]
    tissue_df = tissue_df[tissue_df.columns[args_to_sort]]
    return tissue_df[
        [
            "Terminal Villi",
            "Mature Intermediate Villi",
            "Stem Villi",
            "Villus Sprout",
            "Anchoring Villi",
            "Chorionic Plate",
            "Basal Plate/Septum",
            "Fibrin",
            "Avascular Villi",
        ]
    ]


if __name__ == "__main__":
    typer.run(main)
