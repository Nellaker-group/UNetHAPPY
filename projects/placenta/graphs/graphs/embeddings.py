from pathlib import Path

import torch
import umap
import umap.plot
import numpy as np
from sklearn.cluster import KMeans


def plot_umap_embeddings(
    model, x, edge_index, predictions, plot_name, feature, trained, organ
):
    graph_embeddings = _get_graph_embeddings(model, x, edge_index)
    mapper = _fit_umap(graph_embeddings)
    _plot_umap(mapper, predictions, plot_name, feature, trained, organ)
    return graph_embeddings, mapper


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


def _plot_umap(mapper, predictions, plot_name, feature, trained, organ):
    trained_str = "trained" if trained else "untrained"

    predictions_labelled = np.array([organ.by_id(pred).label for pred in predictions])
    colours_dict = {cell.label: cell.colour for cell in organ.cells}

    print("Plotting UMAP")
    plot = umap.plot.points(mapper, labels=predictions_labelled, color_key=colours_dict)
    save_dir = (
        Path(__file__).absolute().parent.parent.parent
        / "visualisations"
        / "graphs"
        / "graphsage"
        / feature
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    plot_name = f"{trained_str}_{plot_name}.png"
    plot.figure.savefig(save_dir / plot_name)
    print(f"UMAP saved to {save_dir / plot_name}")


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
