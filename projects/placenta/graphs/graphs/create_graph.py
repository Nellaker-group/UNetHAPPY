from torch_cluster import knn_graph, radius_graph
from torch_geometric.transforms import Distance
from scipy.spatial import Voronoi
import matplotlib.tri as tri
import numpy as np


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


# TODO: the edge index should be a Tensor
def make_delaunay_graph(data):
    print(f"Generating delaunay graph")
    triang = tri.Triangulation(data.pos[:, 0], data.pos[:, 1])
    data.edge_index = triang.edges
    print("Graph made!")
    return data
