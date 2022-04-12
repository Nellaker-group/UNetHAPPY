from itertools import product, combinations

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
    plot_example_subgraphs: bool = False,
):
    organ = get_organ(organ_name)
    cell_mapping = {cell.id: cell.label for cell in organ.cells}
    cell_colours = [cell.colour for cell in organ.cells]
    custom_palette = sns.set_palette(sns.color_palette(cell_colours))

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

    # Make list of tile x,y coordinates with width and height
    print(f"Tiling WSI into tiles of size {tile_width}x{tile_height}")
    # Find min x,y and max x,y from points
    xs = data["pos"][:, 0]
    ys = data["pos"][:, 1]
    min_x, min_y = int(xs.min()), int(ys.min())
    max_x, max_y = int(xs.max()), int(ys.max())
    num_columns = int(np.ceil((max_x - min_x) / tile_width))
    num_rows = int(np.ceil((max_y - min_y) / tile_height))
    print(f"rows: {num_rows}, columns: {num_columns}")

    # Make list of minx and miny for each tile
    xy_list = []
    for col in range(num_columns):
        x = min_x + col * tile_width
        xy_list.extend([(x, y) for y in range(min_y, max_y + min_y, tile_height)])

    # Create list of tiles which hold their node indicies from the graph
    print("Extracting nodes within tiles")
    tiles = []
    for i, tile_coord in tqdm(enumerate(xy_list)):
        tile_min_x, tile_min_y = tile_coord[0], tile_coord[1]
        tile_max_x, tile_max_y = tile_min_x + tile_width, tile_min_y + tile_height
        mask = torch.logical_and(
            (torch.logical_and(xs > tile_min_x, (ys > tile_min_y))),
            (torch.logical_and(xs < tile_max_x, (ys < tile_max_y))),
        )
        node_inds = mask.nonzero()[:, 0].tolist()
        # Only save tiles which contain at least one node
        if len(node_inds) > 0:
            tiles.append(
                {
                    "min_x": tile_coord[0],
                    "min_y": tile_coord[1],
                    "node_inds": node_inds,
                    "num_nodes": len(node_inds),
                }
            )
    tiles = pd.DataFrame(tiles)

    # Plot histogram of number of nodes per tile
    plt.figure(figsize=(8, 8))
    sns.displot(tiles, x="num_nodes", binwidth=25)
    plt.savefig("plots/num_nodes_per_tile.png")
    plt.clf()
    plt.close('all')

    # Remove tiles with number of cell points below a min threshold
    tiles_with_too_few_nodes = tiles[tiles.num_nodes < min_cells_in_tile]
    print(f"Removing {len(tiles_with_too_few_nodes)}/{len(tiles)} "
          f"tiles with too few nodes")
    tiles.drop(tiles_with_too_few_nodes.index, inplace=True)
    tiles.reset_index(drop=True, inplace=True)

    # Create a dataset of subgraphs based on the tile nodes
    print("Creating subgraphs within tiles")
    tile_graphs = []
    for i, tile in tqdm(tiles.iterrows()):
        if i > num_tiles:
            break
        node_inds = tile.node_inds
        tile_edge_index, tile_edge_attr = subgraph(
            node_inds, data["edge_index"], data["edge_attr"], relabel_nodes=True
        )
        tile_graph = Data(
            x=data["x"][node_inds],
            edge_index=tile_edge_index,
            edge_attr=tile_edge_attr,
            pos=data["pos"][node_inds],
        )
        tile_graphs.append(tile_graph)

    # Plot some subgraphs
    if plot_example_subgraphs:
        for i, tile_graph in enumerate(tile_graphs):
            visualize_points(
                organ,
                f"plots/subgraphs/tile_{i}.png",
                tile_graph["pos"],
                labels=tile_graph["x"][:, 0].to(torch.int).tolist(),
                edge_index=tile_graph["edge_index"],
                point_size=20,
            )
            plt.clf()
    plt.close('all')

    # Quantify cell types within tile subgraphs
    print("Getting cell types within tiles")
    plt.figure(figsize=(8, 8))
    for i, tile_graph in tqdm(enumerate(tile_graphs)):
        cells_in_graph = tile_graph["x"][:, 0].to(torch.int).tolist()
        cells_df = pd.DataFrame(cells_in_graph, columns=["cell_types"])
        grouped_df = cells_df.groupby(["cell_types"]).size().reset_index(name="counts")
        missing_cells = list(set(cell_mapping.keys()) - set(grouped_df["cell_types"]))
        for cell in missing_cells:
            grouped_df = grouped_df.append(
                pd.Series({"cell_types": cell, "counts": 0}), ignore_index=True
            )
        grouped_df.sort_values("cell_types", inplace=True, ignore_index=True)
        grouped_df["cell_types"] = grouped_df["cell_types"].map(cell_mapping)

        sns.barplot(
            x=grouped_df["cell_types"], y=grouped_df["counts"], palette=custom_palette
        )
        plt.savefig(f"plots/cell_types/tile_{i}.png")
        plt.clf()
    plt.close('all')

    # Quantify cell connections within tile subgraphs
    print("Getting cell types connections within tiles")
    plt.figure(figsize=(12, 8))
    for i, tile_graph in tqdm(enumerate(tile_graphs)):
        edge_index = tile_graph["edge_index"]
        source_cells = tile_graph["x"][:, 0][edge_index[0]].to(torch.int).tolist()
        target_cells = tile_graph["x"][:, 0][edge_index[1]].to(torch.int).tolist()
        connection_df = pd.DataFrame(
            {"source_cells": source_cells, "target_cells": target_cells}
        )
        grouped_connections = (
            connection_df.groupby(["source_cells", "target_cells"])
            .size()
            .reset_index(name="counts")
        )
        # Permute values into canonical order, then further group duplicates
        mask = (
            grouped_connections["source_cells"] <= grouped_connections["target_cells"]
        )
        grouped_connections["source"] = grouped_connections["source_cells"].where(
            mask, grouped_connections["target_cells"]
        )
        grouped_connections["target"] = grouped_connections["target_cells"].where(
            mask, grouped_connections["source_cells"]
        )
        grouped_connections = grouped_connections.groupby(
            ["source", "target"], as_index=False
        )["counts"].sum()

        all_connections = list(combinations(cell_mapping.keys(), 2))
        all_connections.extend([(_, _) for _ in cell_mapping.keys()])
        containing_connections = list(
            zip(
                list(grouped_connections["source"]),
                list(grouped_connections["target"]),
            )
        )
        missing_connections = list(set(all_connections) - set(containing_connections))
        for source, target in missing_connections:
            grouped_connections = grouped_connections.append(
                pd.Series({"source": source, "target": target, "counts": 0}),
                ignore_index=True,
            )
        grouped_connections.sort_values("source", inplace=True, ignore_index=True)

        grouped_connections["source"] = grouped_connections["source"].map(cell_mapping)
        grouped_connections["target"] = grouped_connections["target"].map(cell_mapping)
        grouped_connections["cell_connections"] = (
            grouped_connections["source"] + "-" + grouped_connections["target"]
        )

        plot = sns.barplot(
            x=grouped_connections["cell_connections"],
            y=grouped_connections["counts"],
        )
        plot.set_xticklabels(
            plot.get_xticklabels(),
            rotation=45,
            horizontalalignment="right",
            fontweight="light",
            fontsize=7,
        )
        plt.savefig(f"plots/cell_connections/tile_{i}.png")
        plt.clf()
    plt.close('all')

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
