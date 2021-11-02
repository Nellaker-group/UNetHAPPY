import pandas as pd
from torch_cluster import knn_graph, radius_graph
from torch_geometric.data import Data
from torch_geometric.transforms import Distance
from scipy.spatial import Voronoi
import matplotlib.tri as tri
import numpy as np
import torch

from happy.hdf5.utils import (
    get_embeddings_file,
    get_datasets_in_patch,
    filter_by_confidence,
)


def get_groundtruth_patch(
    organ, project_dir, x_min, y_min, width, height, label_type="full"
):
    ground_truth_df = pd.read_csv(
        project_dir / "results" / "tissue_annots" / "139_tissue_points.tsv", sep="\t"
    )
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


def setup_graph(coords, k, feature, graph_method):
    data = Data(x=torch.Tensor(feature), pos=torch.Tensor(coords.astype("int32")))
    if graph_method == "k":
        graph = make_k_graph(data, k)
    elif graph_method == "delaunay":
        graph = make_delaunay_graph(data)
    else:
        raise ValueError(f"No such graph method: {graph_method}")
    if graph.x.ndim == 1:
        graph.x = graph.x.view(-1, 1)
    return graph


def make_k_graph(data, k):
    print(f"Generating graph for k={k}")
    data.edge_index = knn_graph(data.pos, k=k + 1, loop=True)
    get_edge_distance_weights = Distance(cat=False)
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


# TODO: the edge index and pos should be a Tensor
def make_voronoi_graph(data):
    print(f"Generating voronoi graph")
    vor = Voronoi(data.pos)
    finite_segments = []
    edge_index = []
    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0):
            vertices = vor.vertices[simplex]
            if (
                vertices[:, 0].min() >= vor.min_bound[0]
                and vertices[:, 0].max() <= vor.max_bound[0]
                and vertices[:, 1].min() >= vor.min_bound[1]
                and vertices[:, 1].max() <= vor.max_bound[1]
            ):
                finite_segments.append(vertices)
                edge_index.append(simplex)
    data.pos = np.array(finite_segments).reshape(-1, 2)
    data.edge_index = np.array(edge_index)
    print("Graph made!")
    return data


def make_delaunay_triangulation(data):
    print(f"Generating delaunay triangulation")
    triang = tri.Triangulation(data.pos[:, 0], data.pos[:, 1])
    print("Triangulation made!")
    return triang


def make_delaunay_graph(data):
    print(f"Generating delaunay graph")
    triang = tri.Triangulation(data.pos[:, 0], data.pos[:, 1])
    data.edge_index = torch.tensor(triang.edges.astype("int64"), dtype=torch.long).T
    get_edge_distance_weights = Distance(cat=False)
    data = get_edge_distance_weights(data)
    print("Graph made!")
    return data
