from itertools import combinations

import typer
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from projects.placenta.graphs.graphs.create_graph import (
    get_raw_data,
    setup_graph,
    get_list_of_subgraphs,
)
from projects.placenta.graphs.graphs.enums import MethodArg
from projects.placenta.graphs.analysis.vis_graph_patch import visualize_points
from projects.placenta.graphs.graphs.utils import remove_far_nodes, get_tile_coordinates
from happy.organs.organs import get_organ


# TODO: if you want tile_width in pixels you need to account for the pixel size
def main(
    project_name: str = "placenta",
    organ_name: str = "placenta",
    run_id: int = typer.Option(...),
    x_min: int = 0,
    y_min: int = 0,
    width: int = -1,
    height: int = -1,
    k: int = 6,
    graph_method: MethodArg = MethodArg.k,
    tile_width: int = 2500,
    tile_height: int = 2500,
    min_cells_in_tile: int = 10,
    num_tiles: int = 10,
    plot_subgraphs: bool = False,
    plot_counts: bool = False,
):
    organ = get_organ(organ_name)
    cell_mapping = {cell.id: cell.label for cell in organ.cells}
    cell_colours = {cell.label: cell.colour for cell in organ.cells}
    custom_palette = sns.set_palette(sns.color_palette(list(cell_colours.values())))

    # Get raw data from hdf5 across WSI
    predictions, _, coords, confidence = get_raw_data(
        project_name, run_id, x_min, y_min, width, height
    )
    # Create the graph from the raw data
    data = setup_graph(
        coords, k, predictions, graph_method.value, norm_edges=False, loop=False
    )
    # Remove edges and isolated nodes from graph when edges are too long
    data = remove_far_nodes(data, edge_max_length=25000)

    # Make list of tile x,y coordinates with width and height
    xy_list = get_tile_coordinates(
        data["pos"][:, 0], data["pos"][:, 1], tile_width, tile_height
    )

    # Create a dataset of subgraphs based on tiles
    tile_graphs = get_list_of_subgraphs(
        data,
        xy_list,
        tile_width,
        tile_height,
        min_cells_in_tile,
        max_tiles=num_tiles,
        plot_nodes_per_tile=True,
    )

    # Plot some subgraphs
    if plot_subgraphs:
        for i, tile_graph in enumerate(tqdm(tile_graphs, desc="Plotting subgraphs")):
            visualize_points(
                organ,
                f"plots/subgraphs/tile_{i}.png",
                tile_graph["pos"],
                labels=tile_graph["x"][:, 0].to(torch.int).tolist(),
                edge_index=tile_graph["edge_index"],
                point_size=20,
            )
            plt.close("all")

    # Quantify cell types within tile subgraphs
    plt.figure(figsize=(8, 8))
    for i, tile_graph in enumerate(tqdm(tile_graphs, desc="Cell types in tiles")):
        if plot_counts:
            _plot_cell_counts(tile_graph, cell_mapping, custom_palette, i)
    plt.close("all")

    # Quantify cell connections within tile subgraphs
    all_conns = list(combinations(cell_mapping.keys(), 2))
    all_conns.extend([(_, _) for _ in cell_mapping.keys()])
    all_conns = set(all_conns)
    plt.figure(figsize=(8, 8))
    for i, tile_graph in enumerate(tqdm(tile_graphs, desc="Cell connections in tiles")):
        if plot_counts:
            _plot_cell_connections(tile_graph, all_conns, cell_mapping, cell_colours, i)
    plt.close("all")


def _plot_cell_connections(tile_graph, all_conns, cell_mapping, cell_colours, i):
    edge_index = tile_graph["edge_index"]
    source_cells = tile_graph["x"][:, 0][edge_index[0]].to(torch.int).tolist()
    target_cells = tile_graph["x"][:, 0][edge_index[1]].to(torch.int).tolist()
    connection_df = pd.DataFrame(
        {"source_cells": source_cells, "target_cells": target_cells}
    )
    grouped_conns = (
        connection_df.groupby(["source_cells", "target_cells"])
        .size()
        .reset_index(name="counts")
    )
    # Permute values into canonical order, then further group duplicates
    mask = grouped_conns["source_cells"] <= grouped_conns["target_cells"]
    grouped_conns["source"] = grouped_conns["source_cells"].where(
        mask, grouped_conns["target_cells"]
    )
    grouped_conns["target"] = grouped_conns["target_cells"].where(
        mask, grouped_conns["source_cells"]
    )
    grouped_conns = grouped_conns.groupby(["source", "target"], as_index=False)[
        "counts"
    ].sum()
    # add the missing cell connections with counts of 0 to the whole dataframe
    containing_conns = list(
        zip(list(grouped_conns["source"]), list(grouped_conns["target"]))
    )
    missing_conns = list(all_conns - set(containing_conns))
    missing_conns = pd.DataFrame(missing_conns, columns=["source", "target"])
    missing_conns["counts"] = 0
    grouped_conns = pd.concat([grouped_conns, missing_conns])
    grouped_conns.sort_values("source", inplace=True, ignore_index=True)
    grouped_conns["source"] = grouped_conns["source"].map(cell_mapping)
    grouped_conns["target"] = grouped_conns["target"].map(cell_mapping)
    # plot cell type counts
    heatmap_df = pd.pivot_table(
        grouped_conns, index="source", columns="target", values="counts", fill_value=0
    ).T
    for cell_label in list(cell_mapping.values()):
        if (heatmap_df.loc[cell_label] != 0).sum() > 1 and (
            heatmap_df[cell_label] != 0
        ).sum() == 1:
            heatmap_df[cell_label] = heatmap_df.loc[cell_label]

    heatmap_mask = np.triu(
        np.ones((heatmap_df.shape[0] + 1, heatmap_df.shape[0] + 1), dtype=bool)
    )[:, :-1][1:, :]
    ax = sns.heatmap(
        heatmap_df,
        mask=heatmap_mask,
        annot=True,
        cmap="Blues",
        fmt="g",
        cbar=False,
    )
    colours = list(heatmap_df.index.map(cell_colours))
    for xtick, colour in zip(ax.get_xticklabels(), colours):
        xtick.set_color(colour)
    for ytick, colour in zip(ax.get_yticklabels(), colours):
        ytick.set_color(colour)
    ax.set_ylabel("")
    ax.set_xlabel("")
    plt.savefig(f"plots/cell_connections/tile_{i}.png")
    plt.clf()


def _plot_cell_counts(tile_graph, cell_mapping, custom_palette, i):
    # count cell types in graph
    cells_in_graph = tile_graph["x"][:, 0].to(torch.int).tolist()
    cells_df = pd.DataFrame(cells_in_graph, columns=["cell_types"])
    grouped_df = cells_df.groupby(["cell_types"]).size().reset_index(name="counts")
    # get cell types from organ which are missed from the graph
    missing_cells = list(set(cell_mapping.keys()) - set(grouped_df["cell_types"]))
    missing_cells = pd.DataFrame(missing_cells, columns=["cell_types"])
    missing_cells["counts"] = 0
    # add the missing cell types with counts of 0 to the whole dataframe
    grouped_df = pd.concat([grouped_df, missing_cells])
    grouped_df.sort_values("cell_types", inplace=True, ignore_index=True)
    grouped_df["cell_types"] = grouped_df["cell_types"].map(cell_mapping)
    # plot cell type counts
    sns.barplot(
        x=grouped_df["cell_types"], y=grouped_df["counts"], palette=custom_palette
    )
    plt.savefig(f"plots/cell_types/tile_{i}.png")
    plt.clf()


if __name__ == "__main__":
    typer.run(main)
