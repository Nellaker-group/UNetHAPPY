from itertools import combinations

import typer
import torch
import numpy as np
import pandas as pd
from torch_geometric.utils.isolated import (
    contains_isolated_nodes,
    remove_isolated_nodes,
)
from torch_geometric.utils.subgraph import subgraph
from torch_geometric.data import Data
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from projects.placenta.graphs.graphs.create_graph import get_raw_data, setup_graph
from projects.placenta.graphs.graphs.enums import MethodArg
from projects.placenta.graphs.analysis.vis_graph_patch import visualize_points
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
    data = _remove_far_nodes(data, edge_max_length=25000)
    xs = data["pos"][:, 0]
    ys = data["pos"][:, 1]

    # Make list of tile x,y coordinates with width and height
    xy_list = _get_tile_coordinates(xs, ys, tile_width, tile_height)

    # Create list of tiles which hold their node indicies from the graph
    tiles = []
    for i, tile_coords in enumerate(tqdm(xy_list, desc="Extract nodes within tiles")):
        node_inds = _get_nodes_within_tiles(
            tile_coords, tile_width, tile_height, xs, ys
        )
        # Only save tiles which contain at least one node
        if len(node_inds) > 0:
            tiles.append(
                {
                    "min_x": tile_coords[0],
                    "min_y": tile_coords[1],
                    "node_inds": node_inds,
                    "num_nodes": len(node_inds),
                }
            )
    tiles = pd.DataFrame(tiles)

    # Plot histogram of number of nodes per tile
    _plot_nodes_per_tile(tiles, binwidth=25)

    # Remove tiles with number of cell points below a min threshold
    nodeless_tiles = tiles[tiles.num_nodes < min_cells_in_tile]
    print(f"Removing {len(nodeless_tiles)}/{len(tiles)} tiles with too few nodes")
    tiles.drop(nodeless_tiles.index, inplace=True)
    tiles.reset_index(drop=True, inplace=True)

    # Create a dataset of subgraphs based on the tile nodes
    tiles_node_inds = tiles["node_inds"].to_numpy()
    tile_graphs = make_tile_graph_dataset(tiles_node_inds, data, num_tiles)

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


def make_tile_graph_dataset(tile_node_inds, full_graph, max_tiles):
    tile_graphs = []
    for i, node_inds in enumerate(tqdm(tile_node_inds, desc="Subgraphs within tiles")):
        if i > max_tiles:
            break
        tile_edge_index, tile_edge_attr = subgraph(
            node_inds,
            full_graph["edge_index"],
            full_graph["edge_attr"],
            relabel_nodes=True,
        )
        tile_graph = Data(
            x=full_graph["x"][node_inds],
            edge_index=tile_edge_index,
            edge_attr=tile_edge_attr,
            pos=full_graph["pos"][node_inds],
        )
        tile_graphs.append(tile_graph)
    return tile_graphs


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


def _get_nodes_within_tiles(tile_coords, tile_width, tile_height, all_xs, all_ys):
    tile_min_x, tile_min_y = tile_coords[0], tile_coords[1]
    tile_max_x, tile_max_y = tile_min_x + tile_width, tile_min_y + tile_height
    mask = torch.logical_and(
        (torch.logical_and(all_xs > tile_min_x, (all_ys > tile_min_y))),
        (torch.logical_and(all_xs < tile_max_x, (all_ys < tile_max_y))),
    )
    return mask.nonzero()[:, 0].tolist()


def _plot_nodes_per_tile(tiles, binwidth):
    plt.figure(figsize=(8, 8))
    sns.displot(tiles, x="num_nodes", binwidth=binwidth, color="blue")
    plt.savefig("plots/num_nodes_per_tile.png")
    plt.clf()
    plt.close("all")


def _get_tile_coordinates(all_xs, all_ys, tile_width, tile_height):
    print(f"Tiling WSI into tiles of size {tile_width}x{tile_height}")
    # Find min x,y and max x,y from points and calculate num of rows and cols
    min_x, min_y = int(all_xs.min()), int(all_ys.min())
    max_x, max_y = int(all_xs.max()), int(all_ys.max())
    num_columns = int(np.ceil((max_x - min_x) / tile_width))
    num_rows = int(np.ceil((max_y - min_y) / tile_height))
    print(f"rows: {num_rows}, columns: {num_columns}")
    # Make list of minx and miny for each tile
    xy_list = []
    for col in range(num_columns):
        x = min_x + col * tile_width
        xy_list.extend([(x, y) for y in range(min_y, max_y + min_y, tile_height)])
    return xy_list


# Remove points from graph with edges which are too long (sort by edge length)
def _remove_far_nodes(data, edge_max_length=25000):
    edge_lengths = data["edge_attr"].numpy().ravel()
    sorted_inds_over_length = (np.sort(edge_lengths) > edge_max_length).nonzero()[0]
    bad_edge_inds = np.argsort(data["edge_attr"].numpy().ravel())[
        sorted_inds_over_length
    ]
    print(
        f"Contains isolated nodes before edge removal? "
        f"{contains_isolated_nodes(data['edge_index'], data.num_nodes)}"
    )
    data["edge_index"] = _remove_element_by_indicies(data["edge_index"], bad_edge_inds)
    data["edge_attr"] = _remove_element_by_indicies(data["edge_attr"], bad_edge_inds)
    print(
        f"Contains isolated nodes after edge removal? "
        f"{contains_isolated_nodes(data['edge_index'], data.num_nodes)}"
    )
    print(
        f"Removed {len(bad_edge_inds)} edges "
        f"from graph with edge length > {edge_max_length}"
    )
    data["edge_index"], data["edge_attr"], mask = remove_isolated_nodes(
        data["edge_index"], data["edge_attr"], data.num_nodes
    )
    data["x"] = data["x"][mask]
    data["pos"] = data["pos"][mask]
    print(f"Removed {len(mask[mask == False])} isolated nodes")
    return data


def _remove_element_by_indicies(tensor, inds):
    if len(tensor) > 2:
        mask = torch.ones(tensor.size()[0], dtype=torch.bool)
        mask[inds] = False
        return tensor[mask]
    else:
        mask = torch.ones(tensor.size()[1], dtype=torch.bool)
        mask[inds] = False
        return tensor[:, mask]


if __name__ == "__main__":
    typer.run(main)
