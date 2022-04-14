from itertools import combinations

import typer
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn_extra.cluster import KMedoids
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score

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
    num_clusters: int = 6,
    plot_subgraphs: bool = False,
    plot_counts: bool = False,
    run_over_cell_counts: bool = False,
    run_over_cell_conns: bool = False,
):
    organ = get_organ(organ_name)
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
    num_tiles = len(xy_list) if num_tiles == -1 else num_tiles
    tile_graphs = get_list_of_subgraphs(
        data,
        xy_list,
        tile_width,
        tile_height,
        min_cells_in_tile,
        max_tiles=num_tiles,
        plot_nodes_per_tile=True,
    )

    # Plot selected subgraphs
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
    if run_over_cell_counts:
        cell_counts = np.empty((len(tile_graphs), len(organ.cells)))
        plt.figure(figsize=(8, 8))
        for i, tile_graph in enumerate(tqdm(tile_graphs, desc="Cell types in tiles")):
            cell_counts_df = get_cell_counts(tile_graph, organ)
            cell_counts[i] = np.array(cell_counts_df["counts"])
            if plot_counts:
                _plot_cell_counts(cell_counts_df, organ, i)
        plt.close("all")
        model = KMedoids(metric="euclidean", n_clusters=num_clusters)
        model.fit(cell_counts)
        counts_predictions = model.predict(cell_counts)
        print("")
        print(counts_predictions)

    # Quantify cell connections within tile subgraphs
    if run_over_cell_conns:
        all_conns = _get_all_possible_cell_connections(organ)
        cell_conns = np.empty((len(tile_graphs), len(all_conns)))
        plt.figure(figsize=(8, 8))
        for i, tile_graph in enumerate(tqdm(tile_graphs, desc="Cell conns in tiles")):
            cell_connections_df = get_cell_connections(tile_graph, organ, all_conns)
            cell_conns[i] = np.array(cell_connections_df["counts"])
            if plot_counts:
                _plot_cell_connections(cell_connections_df, organ, i)
        plt.close("all")
        model = KMedoids(metric="euclidean", n_clusters=num_clusters)
        model.fit(cell_conns)
        conns_predictions = model.predict(cell_conns)
        print("")
        print(conns_predictions)

    if run_over_cell_counts and run_over_cell_conns:
        norm_mutual_info_score = normalized_mutual_info_score(
            counts_predictions, conns_predictions
        )
        rand_score = adjusted_rand_score(counts_predictions, conns_predictions)
        print(f"Mutual Information: {norm_mutual_info_score}")
        print(f"Rand Score: {rand_score}")


def get_cell_counts(tile_graph, organ, as_proportion=True):
    cell_mapping = {cell.id: cell.label for cell in organ.cells}
    # count cell types in graph
    cells_in_graph = tile_graph["x"][:, 0].to(torch.int).tolist()
    cells_df = pd.DataFrame(cells_in_graph, columns=["cell_types"])
    grouped_df = cells_df.groupby(["cell_types"]).size().reset_index(name="counts")
    if as_proportion:
        grouped_df["counts"] = grouped_df["counts"].div(
            grouped_df["counts"].sum(), axis=0
        )
    # get cell types from organ which are missed from the graph
    missing_cells = list(set(cell_mapping.keys()) - set(grouped_df["cell_types"]))
    missing_cells = pd.DataFrame(missing_cells, columns=["cell_types"])
    missing_cells["counts"] = 0
    # add the missing cell types with counts of 0 to the whole dataframe
    grouped_df = pd.concat([grouped_df, missing_cells])
    grouped_df.sort_values("cell_types", inplace=True, ignore_index=True)
    grouped_df["cell_types"] = grouped_df["cell_types"].map(cell_mapping)
    return grouped_df


def _plot_cell_counts(cell_counts_df, organ, i):
    cell_colours = [cell.colour for cell in organ.cells]
    custom_palette = sns.set_palette(sns.color_palette(cell_colours))
    # plot cell type counts
    sns.barplot(
        x=cell_counts_df["cell_types"],
        y=cell_counts_df["counts"],
        palette=custom_palette,
    )
    plt.savefig(f"plots/cell_types/tile_{i}.png")
    plt.clf()


def get_cell_connections(tile_graph, organ, all_conns, as_proportion=True):
    cell_mapping = {cell.id: cell.label for cell in organ.cells}
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
    if as_proportion:
        grouped_conns["counts"] = grouped_conns["counts"].div(
            grouped_conns["counts"].sum(), axis=0
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
    return grouped_conns


def _plot_cell_connections(cell_connections_df, organ, i):
    cell_labels = [cell.label for cell in organ.cells]
    cell_colours = {cell.label: cell.colour for cell in organ.cells}
    # make a confusion matrix style structure for the connections
    heatmap_df = pd.pivot_table(
        cell_connections_df,
        index="source",
        columns="target",
        values="counts",
        fill_value=0,
    ).T
    # move all counts to the bottom left triangle of the structure
    for cell_label in list(cell_labels):
        if (heatmap_df.loc[cell_label] != 0).sum() > 1 and (
            heatmap_df[cell_label] != 0
        ).sum() == 1:
            heatmap_df[cell_label] = heatmap_df.loc[cell_label]
    # plot heatmap triangle of cell-cell connections
    heatmap_mask = np.triu(
        np.ones((heatmap_df.shape[0] + 1, heatmap_df.shape[0] + 1), dtype=bool)
    )[:, :-1][1:, :]
    ax = sns.heatmap(
        heatmap_df,
        mask=heatmap_mask,
        annot=True,
        cmap="Blues",
        fmt=".2%",
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


def _get_all_possible_cell_connections(organ):
    cell_ids = [cell.id for cell in organ.cells]
    all_conns = list(combinations(cell_ids, 2))
    all_conns.extend([(_, _) for _ in cell_ids])
    return set(all_conns)


if __name__ == "__main__":
    typer.run(main)
