from pathlib import Path

import torch
import umap
import umap.plot
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from analysis.embeddings.plots import plot_umap


def generate_umap(model, x, edge_index, organ, predictions, run_path, plot_name):
    graph_embeddings, mapper = get_fitted_mapper(model, x, edge_index)
    plot = plot_umap(organ, predictions, mapper)
    print(f"saving umap to {run_path / plot_name}")
    plot.figure.savefig(run_path / plot_name)
    plt.close(plot.figure)
    return graph_embeddings, mapper


def get_fitted_mapper(model, x, edge_index):
    graph_embeddings = _get_graph_embeddings(model, x, edge_index)
    return graph_embeddings, _fit_umap(graph_embeddings)


@torch.no_grad()
def _get_graph_embeddings(model, x, edge_index):
    print("Getting node embeddings for training data")
    model.eval()
    out = model.full_forward(x, edge_index).cpu()
    return out


def _fit_umap(graph_embeddings):
    print("Fitting UMAP to embeddings")
    reducer = umap.UMAP(random_state=42, verbose=True, min_dist=0.1, n_neighbors=15)
    return reducer.fit(graph_embeddings)


def plot_clustering(graph_embeddings, mapper, num_clusters, plot_name, feature):
    kmeans_labels = KMeans(n_clusters=num_clusters).fit_predict(graph_embeddings)

    plot = umap.plot.points(mapper, labels=kmeans_labels)
    save_dir = (
        Path(__file__).absolute().parent.parent.parent
        / "visualisations"
        / "graphs"
        / "graphsage"
        / feature
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    plot_name = f"clustered_{plot_name}.png"
    plot.figure.savefig(save_dir / plot_name)
    print(f"Clustered UMAP saved to {save_dir / plot_name}")
    return kmeans_labels
