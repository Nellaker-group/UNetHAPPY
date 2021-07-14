from torch_cluster import knn_graph, radius_graph
from torch_geometric.transforms import Distance
from scipy.spatial import Voronoi, Delaunay


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


def make_voronoi_graph(data):
    print(f"Generating voronoi diagram")
    graph = Voronoi(data.pos)
    print("Graph made!")
    return graph

def make_delaunay_graph(data):
    print(f"Generating delaunay diagram")
    graph = Delaunay(data.pos)
    print("Graph made!")
    return graph
