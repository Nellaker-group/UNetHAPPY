from pathlib import Path

import typer
import torch
import pandas as pd
import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score, adjusted_mutual_info_score

from happy.utils.utils import get_device
from happy.utils.utils import get_project_dir
from happy.organs.organs import get_organ
from graphs.graphs.create_graph import get_raw_data, setup_graph
from graphs.graphs.embeddings import (
    get_graph_embeddings,
    fit_umap,
    plot_graph_umap,
    plot_clustering,
    fit_clustering,
)
from graphs.graphs.utils import get_feature
from graphs.graphs.enums import FeatureArg, MethodArg
from graphs.analysis.vis_graph_patch import visualize_points


def main(
    project_name: str = "placenta",
    organ_name: str = "placenta",
    exp_name: str = typer.Option(...),
    model_weights_dir: str = typer.Option(...),
    run_id: int = typer.Option(...),
    x_min: int = 0,
    y_min: int = 0,
    width: int = -1,
    height: int = -1,
    k: int = 6,
    feature: FeatureArg = FeatureArg.embeddings,
    top_conf: bool = False,
    graph_method: MethodArg = MethodArg.k,
    num_clusters: int = 3,
    clustering_method: str = "kmeans",
):
    device = get_device()
    project_dir = get_project_dir(project_name)
    organ = get_organ(organ_name)

    # Get data from hdf5 files
    predictions, embeddings, coords, confidence = get_raw_data(
        project_name, run_id, x_min, y_min, width, height, top_conf
    )
    feature_data = get_feature(feature.value, predictions, embeddings)
    data = setup_graph(coords, k, feature_data, graph_method.value)
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)

    # TODO: the coords don't match with xs and ys for some reason... Fix this
    # Get ground truth manually annotated data
    xs, ys, tissue_class = get_ground_truth_patch(
        organ, project_dir, x_min, y_min, width, height
    )

    # Setup trained model
    pretrained_path = (
        project_dir
        / "results"
        / "graph"
        / exp_name
        / model_weights_dir
        / "graph_model.pt"
    )
    model = torch.load(pretrained_path, map_location=device)

    # Setup paths
    save_path = Path(*pretrained_path.parts[:-1]) / "eval"
    save_path.mkdir(exist_ok=True)
    conf_str = "_top_conf" if top_conf else ""
    plot_name = f"x{x_min}_y{y_min}_w{width}_h{height}{conf_str}"

    # fit and plot umap using graph embeddings of the trained model
    graph_embeddings = get_graph_embeddings(model, x, edge_index)
    fitted_umap = fit_umap(graph_embeddings)
    plot_graph_umap(organ, predictions, fitted_umap, save_path, f"eval_{plot_name}.png")

    save_path = Path(*save_path.parts) / clustering_method / f"{num_clusters}_clusters"
    save_path.mkdir(parents=True, exist_ok=True)

    # Fit a clustering method on the graph embeddings
    cluster_labels = fit_clustering(
        num_clusters, graph_embeddings, clustering_method, fitted_umap
    )

    # Evaluate against ground truth tissue annotations
    rand_score = adjusted_rand_score(tissue_class, cluster_labels)
    mutual_info_score = adjusted_mutual_info_score(tissue_class, cluster_labels)
    print(f"Rand score: {rand_score}")
    print(f"Matual info score: {mutual_info_score}")

    plot_clustering(fitted_umap, plot_name, save_path, cluster_labels)

    visualize_points(
        organ,
        save_path / f"patch_{plot_name}.png",
        data.pos,
        labels=data.x,
        edge_index=data.edge_index,
        edge_weight=data.edge_attr,
        colours=cluster_labels,
    )


def get_ground_truth_patch(organ, project_dir, x_min, y_min, width, height):
    ground_truth_df = pd.read_csv(
        project_dir / "results" / "tissue_annots" / "139_tissue_points.csv"
    )
    xs = ground_truth_df["px"].to_numpy()
    ys = ground_truth_df["py"].to_numpy()
    tissue_classes = ground_truth_df["class"].to_numpy()

    if x_min == 0 and y_min == 0 and width == -1 and height == -1:
        tissue_ids = np.array(
            [organ.tissue_by_label(tissue_name).id for tissue_name in tissue_classes])
        return xs, ys, tissue_ids

    mask = np.logical_and(
        (np.logical_and(xs > x_min, (ys > y_min))),
        (np.logical_and(xs < (x_min + width), (ys < (y_min + height)))),
    )
    patch_xs = xs[mask]
    patch_ys = ys[mask]
    patch_tissue_classes = tissue_classes[mask]

    patch_tissue_ids = np.array(
        [organ.tissue_by_label(tissue_name).id for tissue_name in patch_tissue_classes])
    return patch_xs, patch_ys, patch_tissue_ids


if __name__ == "__main__":
    typer.run(main)
