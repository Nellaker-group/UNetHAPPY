import os

import pandas as pd
from torch_cluster import radius_graph
from torch_geometric.data import Data
from torch_geometric.utils.subgraph import subgraph
from torch_geometric.transforms import Distance, KNNGraph
from scipy.spatial import Voronoi
from tqdm import tqdm
import matplotlib.tri as tri
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch

from happy.hdf5.utils import (
    get_embeddings_file,
    get_datasets_in_patch,
    filter_by_confidence,
)
from projects.placenta.graphs.analysis.knot_nuclei_to_point import process_knt_cells


def get_groundtruth_patch(
    organ, project_dir, x_min, y_min, width, height, tissue_label_tsv, label_type="full"
):
    if not tissue_label_tsv:
        print("No tissue label tsv supplied")
        return None, None, None
    tissue_label_path = project_dir / "results" / "tissue_annots" / tissue_label_tsv
    if not os.path.exists(str(tissue_label_path)):
        print("No tissue label tsv found")
        return None, None, None

    ground_truth_df = pd.read_csv(tissue_label_path, sep="\t")
    xs = ground_truth_df["px"].to_numpy()
    ys = ground_truth_df["py"].to_numpy()
    tissue_classes = ground_truth_df["class"].to_numpy()

    if x_min == 0 and y_min == 0 and width == -1 and height == -1:
        sort_args = np.lexsort((ys, xs))
        tissue_ids = np.array(
            [
                _get_id_by_tissue_name(organ, tissue_name, label_type)
                for tissue_name in tissue_classes
            ]
        )
        return xs[sort_args], ys[sort_args], tissue_ids[sort_args]

    mask = np.logical_and(
        (np.logical_and(xs > x_min, (ys > y_min))),
        (np.logical_and(xs < (x_min + width), (ys < (y_min + height)))),
    )
    patch_xs = xs[mask]
    patch_ys = ys[mask]
    patch_tissue_classes = tissue_classes[mask]

    patch_tissue_ids = np.array(
        [
            _get_id_by_tissue_name(organ, tissue_name, label_type)
            for tissue_name in patch_tissue_classes
        ]
    )
    sort_args = np.lexsort((patch_ys, patch_xs))

    return patch_xs[sort_args], patch_ys[sort_args], patch_tissue_ids[sort_args]


def _get_id_by_tissue_name(organ, tissue_name, id_type):
    if id_type == "full":
        tissue_id = organ.tissue_by_label(tissue_name).id
    elif id_type == "alt":
        tissue_id = organ.tissue_by_label(tissue_name).alt_id
    else:
        tissue_id = organ.tissue_by_label(tissue_name).tissue_id
    return tissue_id


def get_raw_data(project_name, run_id, x_min, y_min, width, height, top_conf=False):
    embeddings_path = get_embeddings_file(project_name, run_id)
    print(f"Getting data from: {embeddings_path}")
    print(f"Using patch of size: x{x_min}, y{y_min}, w{width}, h{height}")
    # Get hdf5 datasets contained in specified box/patch of WSI
    predictions, embeddings, coords, confidence = get_datasets_in_patch(
        embeddings_path, x_min, y_min, width, height
    )
    if top_conf:
        predictions, embeddings, coords, confidence = filter_by_confidence(
            predictions, embeddings, coords, confidence, 0.9, 1.0
        )
    print(f"Data loaded with {len(predictions)} nodes")
    sort_args = np.lexsort((coords[:, 1], coords[:, 0]))
    coords = coords[sort_args]
    predictions = predictions[sort_args]
    embeddings = embeddings[sort_args]
    confidence = confidence[sort_args]
    print("Data sorted by x coordinates")

    return predictions, embeddings, coords, confidence

def process_knts(organ, predictions, embeddings, coords, confidence, tissues=None):
    # Turn isolated knts into syn and group large knts into one point
    (
        predictions,
        embeddings,
        coords,
        confidence,
        inds_to_remove,
    ) = process_knt_cells(
        predictions, embeddings, coords, confidence, organ, 50, 3, plot=False
    )
    # Remove points from tissue ground truth as well
    if tissues is not None:
        tissues = np.delete(tissues, inds_to_remove, axis=0)
    return predictions, embeddings, coords, confidence, tissues


def setup_graph(coords, k, feature, graph_method, norm_edges=True, loop=True):
    data = Data(x=torch.Tensor(feature), pos=torch.Tensor(coords.astype("int32")))
    if graph_method == "k":
        graph = make_k_graph(data, k, norm_edges, loop)
    elif graph_method == "delaunay":
        graph = make_delaunay_graph(data, norm_edges)
    elif graph_method == "intersection":
        graph = make_intersection_graph(data, k, norm_edges)
    else:
        raise ValueError(f"No such graph method: {graph_method}")
    if graph.x.ndim == 1:
        graph.x = graph.x.view(-1, 1)
    return graph


def make_k_graph(data, k, norm_edges=True, loop=True):
    print(f"Generating graph for k={k}")
    data = KNNGraph(k=k + 1, loop=loop, force_undirected=True)(data)
    get_edge_distance_weights = Distance(cat=False, norm=norm_edges)
    data = get_edge_distance_weights(data)
    print(f"Graph made with {len(data.edge_index[0])} edges!")
    return data


def make_radius_k_graph(data, radius, k):
    print(f"Generating graph for radius={radius} and k={k}")
    data.edge_index = radius_graph(data.pos, r=radius, max_num_neighbors=k)
    print("Graph made!")
    return data


def make_voronoi(data):
    print(f"Generating voronoi diagram")
    vor = Voronoi(data.pos)
    print("Voronoi made!")
    return vor


def make_delaunay_triangulation(data):
    print(f"Generating delaunay triangulation")
    triang = tri.Triangulation(data.pos[:, 0], data.pos[:, 1])
    print("Triangulation made!")
    return triang


def make_delaunay_graph(data, norm_edges=True):
    print(f"Generating delaunay graph")
    triang = tri.Triangulation(data.pos[:, 0], data.pos[:, 1])
    data.edge_index = torch.tensor(triang.edges.astype("int64"), dtype=torch.long).T
    get_edge_distance_weights = Distance(cat=False, norm=norm_edges)
    data = get_edge_distance_weights(data)
    print(f"Graph made with {len(data.edge_index[0])} edges!")
    return data


def make_intersection_graph(data, k, norm_edges=True):
    print(f"Generating graph for k={k}")
    knn_graph = KNNGraph(k=k + 1, loop=False, force_undirected=True)(data)
    knn_edge_index = knn_graph.edge_index.T
    knn_edge_index = np.array(knn_edge_index.tolist())
    print(f"Generating delaunay graph")
    triang = tri.Triangulation(data.pos[:, 0], data.pos[:, 1])
    delaunay_edge_index = triang.edges.astype("int64")

    print(f"Generating intersection of both graphs")
    _, ncols = knn_edge_index.shape
    dtype = ", ".join([str(knn_edge_index.dtype)] * ncols)
    intersection = np.intersect1d(
        knn_edge_index.view(dtype), delaunay_edge_index.view(dtype)
    )
    intersection = intersection.view(knn_edge_index.dtype).reshape(-1, ncols)
    intersection = torch.tensor(intersection, dtype=torch.long).T
    data.edge_index = intersection

    get_edge_distance_weights = Distance(cat=False, norm=norm_edges)
    data = get_edge_distance_weights(data)
    print(f"Graph made with {len(data.edge_index[0])} edges!")
    return data


def get_list_of_subgraphs(
    data,
    tile_coordinates,
    tile_width,
    tile_height,
    min_cells_in_tile,
    max_tiles=-1,
    plot_nodes_per_tile=False,
):
    # Create list of tiles which hold their node indicies from the graph
    tiles = []
    for i, tile_coords in enumerate(
        tqdm(tile_coordinates, desc="Extract nodes within tiles")
    ):
        node_inds = get_nodes_within_tiles(
            tile_coords, tile_width, tile_height, data["pos"][:, 0], data["pos"][:, 1]
        )
        tiles.append(
            {
                "tile_index": i,
                "min_x": tile_coords[0],
                "min_y": tile_coords[1],
                "node_inds": node_inds,
                "num_nodes": len(node_inds),
            }
        )
    tiles = pd.DataFrame(tiles)

    # Plot histogram of number of nodes per tile
    if plot_nodes_per_tile:
        _plot_nodes_per_tile(tiles, binwidth=25)

    # Remove tiles with number of cell points below a min threshold
    nodeless_tiles = tiles[tiles.num_nodes < min_cells_in_tile]
    print(f"Removing {len(nodeless_tiles)}/{len(tiles)} tiles with too few nodes")
    tiles.drop(nodeless_tiles.index, inplace=True)
    tiles.reset_index(drop=True, inplace=True)

    # Create a dataset of subgraphs based on the tile nodes
    tiles_node_inds = tiles["node_inds"].to_numpy()
    removed_tiles = list(nodeless_tiles.index)
    return make_tile_graph_dataset(tiles_node_inds, data, max_tiles), removed_tiles


def get_nodes_within_tiles(tile_coords, tile_width, tile_height, all_xs, all_ys):
    tile_min_x, tile_min_y = tile_coords[0], tile_coords[1]
    tile_max_x, tile_max_y = tile_min_x + tile_width, tile_min_y + tile_height
    if isinstance(all_xs, torch.Tensor) and isinstance(all_ys, torch.Tensor):
        mask = torch.logical_and(
            (torch.logical_and(all_xs > tile_min_x, (all_ys > tile_min_y))),
            (torch.logical_and(all_xs < tile_max_x, (all_ys < tile_max_y))),
        )
        return mask.nonzero()[:, 0].tolist()
    else:
        mask = np.logical_and(
            (np.logical_and(all_xs > tile_min_x, (all_ys > tile_min_y))),
            (np.logical_and(all_xs < tile_max_x, (all_ys < tile_max_y))),
        )
        return mask.nonzero()[0].tolist()


def _plot_nodes_per_tile(tiles, binwidth):
    plt.figure(figsize=(8, 8))
    sns.displot(tiles, x="num_nodes", binwidth=binwidth, color="blue")
    plt.savefig("plots/num_nodes_per_tile.png")
    plt.clf()
    plt.close("all")


def make_tile_graph_dataset(tile_nodes, full_graph, max_tiles):
    tile_graphs = []
    for i, node_inds in enumerate(tqdm(tile_nodes, desc="Get subgraphs within tiles")):
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
