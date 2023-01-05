from itertools import combinations

import typer
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.cm import get_cmap
import seaborn as sns
from sklearn_extra.cluster import KMedoids
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score

from happy.graph.create_graph import (
    get_raw_data,
    setup_graph,
    get_list_of_subgraphs,
)
from projects.placenta.graphs.graphs.enums import MethodArg
from projects.placenta.graphs.analysis.vis_graph_patch import visualize_points
from projects.placenta.graphs.graphs.utils import remove_far_nodes, get_tile_coordinates
from happy.organs import get_organ
from happy.utils.utils import get_project_dir


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

    project_dir = get_project_dir(project_name)
    save_dir = project_dir / "results" / "patch_graph" / f"run_{run_id}"
    save_dir.mkdir(parents=True, exist_ok=True)
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

    # Create a datasets of subgraphs based on tiles
    num_tiles = len(xy_list) if num_tiles == -1 else num_tiles
    tile_graphs, nodeless_tiles = get_list_of_subgraphs(
        data,
        xy_list,
        tile_width,
        tile_height,
        min_cells_in_tile,
        max_tiles=num_tiles,
        plot_nodes_per_tile=True,
    )
    xy_list = np.delete(xy_list, nodeless_tiles, axis=0)[:num_tiles]

    # Plot selected subgraphs
    if plot_subgraphs:
        subgraph_save_dir = save_dir / "subgraphs"
        subgraph_save_dir.mkdir(parents=True, exist_ok=True)
        for i, tile_graph in enumerate(tqdm(tile_graphs, desc="Plotting subgraphs")):
            save_path = subgraph_save_dir / f"tile_{i}.png"
            visualize_points(
                organ,
                save_path,
                tile_graph["pos"],
                labels=tile_graph["x"][:, 0].to(torch.int).tolist(),
                edge_index=tile_graph["edge_index"],
                point_size=20,
            )
            plt.close("all")

    # Quantify cell types within tile subgraphs
    if run_over_cell_counts:
        counts_save_dir = save_dir / "cell_types"
        counts_save_dir.mkdir(parents=True, exist_ok=True)
        wsi_save_dir = save_dir / "wsi"
        wsi_save_dir.mkdir(parents=True, exist_ok=True)
        cell_counts = np.empty((len(tile_graphs), len(organ.cells)))
        plt.figure(figsize=(8, 8))
        for i, tile_graph in enumerate(tqdm(tile_graphs, desc="Cell types in tiles")):
            cell_counts_df = get_cell_counts(tile_graph, organ)
            cell_counts[i] = np.array(cell_counts_df["counts"])
            if plot_counts:
                _plot_cell_counts(cell_counts_df, organ, i, counts_save_dir)
        plt.close("all")
        model = KMedoids(metric="euclidean", n_clusters=num_clusters).fit(cell_counts)
        counts_predictions = model.predict(cell_counts)
        # Visualise patch labels into the shape of the WSI
        visualize_patches(
            xy_list,
            tile_width,
            tile_height,
            counts_predictions,
            wsi_save_dir / "count_patches.png"
        )
        plt.figure(figsize=(8, 8))
        # Get and save medoid histograms and subgraphs
        medoids_save_dir = save_dir / "medoids"
        medoids_save_dir.mkdir(parents=True, exist_ok=True)
        medoids = model.cluster_centers_
        for i, medoid in enumerate(medoids):
            cell_types = [cell.label for cell in organ.cells]
            _plot_cell_counts(
                pd.DataFrame({"cell_types": cell_types, "counts": medoid}),
                organ,
                i,
                medoids_save_dir,
            )
        plt.close("all")
        medoid_subgraph_save_dir = save_dir / "medoids"
        medoid_subgraph_save_dir.mkdir(parents=True, exist_ok=True)
        medoid_inds = model.medoid_indices_
        medoid_subgraphs = np.array(tile_graphs, dtype=object)[medoid_inds]
        for i, medoid_subgraph in enumerate(medoid_subgraphs):
            save_path = medoid_subgraph_save_dir / f"subgraph_{i}.png"
            visualize_points(
                organ,
                save_path,
                medoid_subgraph[3][1],
                labels=medoid_subgraph[0][1][:, 0].to(torch.int).tolist(),
                edge_index=medoid_subgraph[1][1],
                point_size=20,
            )
            plt.close("all")

    # Quantify cell connections within tile subgraphs
    if run_over_cell_conns:
        conns_save_dir = save_dir / "cell_connections"
        conns_save_dir.mkdir(parents=True, exist_ok=True)
        wsi_save_dir = save_dir / "wsi"
        wsi_save_dir.mkdir(parents=True, exist_ok=True)
        all_conns = _get_all_possible_cell_connections(organ)
        cell_conns = np.empty((len(tile_graphs), len(all_conns)))
        plt.figure(figsize=(8, 8))
        for i, tile_graph in enumerate(tqdm(tile_graphs, desc="Cell conns in tiles")):
            cell_connections_df = get_cell_connections(tile_graph, organ, all_conns)
            cell_conns[i] = np.array(cell_connections_df["counts"])
            if plot_counts:
                _plot_cell_connections(cell_connections_df, organ, i, conns_save_dir)
        plt.close("all")
        model = KMedoids(metric="euclidean", n_clusters=num_clusters).fit(cell_conns)
        conns_predictions = model.predict(cell_conns)
        visualize_patches(
            xy_list,
            tile_width,
            tile_height,
            conns_predictions,
            wsi_save_dir / "conns_patches.png"
        )

    # Compare label permutation invariant agreement between cluster assignments
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


def _plot_cell_counts(cell_counts_df, organ, i, save_dir):
    cell_colours = [cell.colour for cell in organ.cells]
    custom_palette = sns.set_palette(sns.color_palette(cell_colours))
    # plot cell type counts
    sns.barplot(
        x=cell_counts_df["cell_types"],
        y=cell_counts_df["counts"],
        palette=custom_palette,
    )
    save_path = save_dir / f"tile_{i}.png"
    plt.savefig(save_path)
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


def _plot_cell_connections(cell_connections_df, organ, i, save_path):
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
    plt.savefig(save_path / f"tile_{i}.png")
    plt.clf()


def _get_all_possible_cell_connections(organ):
    cell_ids = [cell.id for cell in organ.cells]
    all_conns = list(combinations(cell_ids, 2))
    all_conns.extend([(_, _) for _ in cell_ids])
    return set(all_conns)


def visualize_patches(
    tile_coordinates, tile_width, tile_height, tile_labels, save_path
):
    cmap_colours = get_cmap("tab10").colors
    keys = list(range(13))
    colourmap = dict(zip(keys, cmap_colours))

    min_x, min_y = tile_coordinates[:, 0].min(), tile_coordinates[:, 1].min()
    max_x = tile_coordinates[:, 0].max() + tile_width
    max_y = tile_coordinates[:, 1].max() + tile_height

    if max_x > max_y:
        max_y = max_x
    else:
        max_x = max_y

    fig, ax = plt.subplots(1, figsize=(8, 8))
    for i, tile in enumerate(tile_coordinates):
        x, y = tile[0], tile[1]
        rect = patches.Rectangle(
            (x, y),
            tile_width,
            tile_height,
            alpha=0.7,
            facecolor=colourmap[tile_labels[i]],
        )
        ax.add_patch(rect)
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    plt.gca().invert_yaxis()
    plt.axis("off")
    fig.tight_layout()
    plt.savefig(save_path)



if __name__ == "__main__":
    typer.run(main)
