import torch
import umap
import umap.plot
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt

from analysis.embeddings.plots import plot_umap


def plot_graph_umap(organ, predictions, mapper, save_dir, plot_name):
    plot = plot_umap(organ, predictions, mapper)
    print(f"saving umap to {save_dir / plot_name}")
    plot.figure.savefig(save_dir / plot_name)
    plt.close(plot.figure)


@torch.no_grad()
def get_graph_embeddings(model, x, edge_index):
    print("Getting node embeddings for training data")
    model.eval()
    out = model.full_forward(x, edge_index).cpu()
    return out


def fit_umap(graph_embeddings):
    print("Fitting UMAP to embeddings")
    reducer = umap.UMAP(random_state=42, verbose=True, min_dist=0.0, n_neighbors=30)
    return reducer.fit(graph_embeddings)


def fit_clustering(num_clusters, graph_embeddings, clustering_method, mapper=None):
    if clustering_method == "kmeans":
        labels = KMeans(n_clusters=num_clusters).fit_predict(graph_embeddings)
    elif clustering_method == 'dbscan':
        labels = DBSCAN(eps=0.1).fit_predict(graph_embeddings)
    elif clustering_method == 'umap':
        umap_embeddings = mapper.transform(graph_embeddings)
        labels = KMeans(n_clusters=num_clusters).fit_predict(umap_embeddings)
    else:
        raise ValueError(f"No such clustering method: {clustering_method}")
    return labels


def plot_clustering(mapper, plot_name, save_dir, cluster_labels):
    plot = umap.plot.points(mapper, labels=cluster_labels)
    plot_name = f"clustered_{plot_name}.png"
    plot.figure.savefig(save_dir / plot_name)
    print(f"Clustered UMAP saved to {save_dir / plot_name}")
