from pathlib import Path
import os

import typer
import torch
from sklearn.metrics.cluster import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
    fowlkes_mallows_score,
    homogeneity_completeness_v_measure,
)

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
from graphs.graphs.create_graph import get_groundtruth_patch


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
    model_type: str = "graphsage",
    graph_method: MethodArg = MethodArg.k,
    num_clusters: int = 3,
    clustering_method: str = "kmeans",
    plot_umap: bool = True,
    remove_unlabelled: bool = True,
    alt_groundtruth: bool = False,
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
    pos = data.pos

    # Get ground truth manually annotated data
    xs, ys, tissue_class = get_groundtruth_patch(
        organ, project_dir, x_min, y_min, width, height, alt_groundtruth
    )

    # Setup trained model
    pretrained_path = (
        project_dir
        / "results"
        / "graph"
        / model_type
        / exp_name
        / model_weights_dir
        / "graph_model.pt"
    )
    model = torch.load(pretrained_path, map_location=device)

    # Setup paths
    save_path = Path(*pretrained_path.parts[:-1]) / "eval"
    save_path.mkdir(exist_ok=True)
    cluster_save_path = save_path / clustering_method / f"{num_clusters}_clusters"
    cluster_save_path.mkdir(parents=True, exist_ok=True)
    conf_str = "_top_conf" if top_conf else ""
    plot_name = f"x{x_min}_y{y_min}_w{width}_h{height}{conf_str}"

    # Get graph embeddings of trained model on patch
    graph_embeddings = get_graph_embeddings(model_type, model, x, edge_index)

    # fit and plot umap with cell classes
    if plot_umap:
        fitted_umap = fit_umap(graph_embeddings)
        plot_graph_umap(
            organ, predictions, fitted_umap, save_path, f"eval_{plot_name}.png"
        )

    # Fit a clustering method on the graph embeddings
    mapper = fitted_umap if clustering_method == 'umap' else None
    cluster_labels = fit_clustering(
        num_clusters, graph_embeddings, clustering_method, mapper=mapper
    )

    # Plot the cluster labels onto the umap of the graph embeddings
    if plot_umap:
        plot_clustering(fitted_umap, plot_name, cluster_save_path, cluster_labels)

    # Remove unlabelled (class 0) ground truth points
    if remove_unlabelled:
        unlabelled_inds = tissue_class.nonzero()
        tissue_class = tissue_class[unlabelled_inds]
        cluster_labels = cluster_labels[unlabelled_inds]
        pos = pos[unlabelled_inds]

    # Evaluate against ground truth tissue annotations
    if run_id == 16:
        evaluate(tissue_class, cluster_labels)

    # Visualise cluster labels on graph patch
    print("Generating image")
    visualize_points(
        organ,
        cluster_save_path / f"patch_{plot_name}.png",
        pos,
        colours=cluster_labels,
        width=width,
        height=height,
    )


def evaluate(tissue_class, cluster_labels):
    adj_rand_score = adjusted_rand_score(tissue_class, cluster_labels)
    norm_mutual_info_score = normalized_mutual_info_score(tissue_class, cluster_labels)
    adj_mutual_info_score = adjusted_mutual_info_score(tissue_class, cluster_labels)
    fm_score = fowlkes_mallows_score(tissue_class, cluster_labels)
    hcv_scores = list(homogeneity_completeness_v_measure(tissue_class, cluster_labels))
    hcv_scores = [round(score, 3) for score in hcv_scores]
    print("-----------------------")
    print(f"Adjusted rand score: {adj_rand_score:.3f}")
    print(f"Normalised mutual info score: {norm_mutual_info_score:.3f}")
    print(f"Adjusted mutual info score: {adj_mutual_info_score:.3f}")
    print(f"Fowlkes Mallows score: {fm_score:.3f}")
    print(f"Homgeneity | Completeness | V-Measure: {hcv_scores}")
    print("-----------------------")


if __name__ == "__main__":
    # exp_name = "new_k_diff_layers_all"
    # run_id = 16
    # graph_method = MethodArg.k
    # x_min = 41203
    # y_min = 21344
    # width = 15000
    # height = 15000
    # num_clusters = 4
    # clustering_method = "kmeans"
    #
    # project_dir = get_project_dir("placenta")
    # exp_dir = project_dir / "results" / "graph" / exp_name
    #
    # for dir in os.listdir(exp_dir):
    #     if os.path.isdir(os.path.join(exp_dir, dir)):
    #         model_weights_dir = dir
    #         print(f"Getting model from: {model_weights_dir}")
    #
    #         main(
    #             exp_name=exp_name,
    #             model_weights_dir=model_weights_dir,
    #             run_id=run_id,
    #             graph_method=graph_method,
    #             x_min=x_min,
    #             y_min=y_min,
    #             width=width,
    #             height=height,
    #             num_clusters=num_clusters,
    #             clustering_method=clustering_method,
    #             plot_umap=False,
    #         )

    typer.run(main)
