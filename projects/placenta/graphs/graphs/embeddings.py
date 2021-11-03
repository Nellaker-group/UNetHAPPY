import torch
import umap
import umap.plot
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
import numpy as np

from analysis.embeddings.plots import plot_cell_umap


def plot_cell_graph_umap(organ, predictions, mapper, save_dir, plot_name):
    plot = plot_cell_umap(organ, predictions, mapper)
    print(f"saving umap to {save_dir / plot_name}")
    plot.figure.savefig(save_dir / plot_name)
    plt.close(plot.figure)


@torch.no_grad()
def get_graph_embeddings(model_type, model, x, edge_index):
    print("Generating node embeddings")
    model.eval()
    if model_type == "graphsage":
        out = model.full_forward(x, edge_index).cpu()
    elif model_type == "infomax":
        out = model.encoder.full_forward(x, edge_index).cpu()
    else:
        raise ValueError(f"No such model type implemented: {model_type}")
    return out


def fit_umap(graph_embeddings):
    print("Fitting UMAP to embeddings")
    reducer = umap.UMAP(random_state=42, verbose=True, min_dist=0.0, n_neighbors=30)
    return reducer.fit(graph_embeddings)


def fit_clustering(num_clusters, graph_embeddings, clustering_method, mapper=None):
    if clustering_method == "kmeans":
        cluster_method = KMeans(n_clusters=num_clusters).fit(graph_embeddings)
    elif clustering_method == "dbscan":
        cluster_method = DBSCAN(eps=0.1).fit(graph_embeddings)
    elif clustering_method == "umap":
        graph_embeddings = mapper.transform(graph_embeddings)
        cluster_method = KMeans(n_clusters=num_clusters).fit(graph_embeddings)
    else:
        raise ValueError(f"No such clustering method: {clustering_method}")
    labels = cluster_method.predict(graph_embeddings)
    return labels, cluster_method


def plot_tissue_umap(organ, mapper, plot_name, save_dir, cluster_labels):
    # Note: this only works for 9-label tissue ids
    label_names = np.array(
        [organ.tissues[pred].label for pred in cluster_labels]
    )
    plot = umap.plot.points(mapper, labels=label_names)
    plot_name = f"labelled_{plot_name}.png"
    plot.figure.savefig(save_dir / plot_name)
    print(f"Clustered UMAP saved to {save_dir / plot_name}")
    plt.close(plot.figure)
