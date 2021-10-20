from pathlib import Path

import typer
import torch
from sklearn.metrics import accuracy_score, top_k_accuracy_score, f1_score
import numpy as np
from torch_geometric.data import NeighborSampler

from happy.utils.utils import get_device
from happy.utils.utils import get_project_dir
from happy.organs.organs import get_organ
from graphs.graphs.create_graph import get_raw_data, setup_graph
from graphs.graphs.embeddings import fit_umap, plot_graph_umap, plot_labels_on_umap
from graphs.graphs.utils import get_feature
from graphs.graphs.enums import FeatureArg, MethodArg
from graphs.analysis.vis_graph_patch import visualize_points
from graphs.graphs.create_graph import get_groundtruth_patch

np.random.seed(2)


def main(
    project_name: str = "placenta",
    organ_name: str = "placenta",
    exp_name: str = typer.Option(...),
    model_weights_dir: str = typer.Option(...),
    model_name: str = typer.Option(...),
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
        / model_name
    )
    model = torch.load(pretrained_path, map_location=device)
    model_epochs = (
        "model_final"
        if model_name == "graph_model.pt"
        else f"model_{model_name.split('_')[0]}"
    )

    # Setup paths
    save_path = Path(*pretrained_path.parts[:-1]) / "eval" / model_epochs
    save_path.mkdir(parents=True, exist_ok=True)
    conf_str = "_top_conf" if top_conf else ""
    plot_name = f"x{x_min}_y{y_min}_w{width}_h{height}{conf_str}"

    # Dataloader for eval, feeds in whole graph
    eval_loader = NeighborSampler(
        data.edge_index,
        node_idx=None,
        sizes=[-1],
        batch_size=512,
        shuffle=False,
    )

    # Run inference and get predicted labels for nodes
    model.eval()
    out, embeddings = model.inference(x, eval_loader, device)
    predicted_labels = out.argmax(dim=-1, keepdim=True).squeeze()

    if plot_umap:
        # fit and plot umap with cell classes
        fitted_umap = fit_umap(embeddings)
        plot_graph_umap(
            organ, predictions, fitted_umap, save_path, f"eval_{plot_name}.png"
        )
        # Plot the predicted labels onto the umap of the graph embeddings
        plot_labels_on_umap(fitted_umap, plot_name, save_path, predicted_labels)
        plot_labels_on_umap(fitted_umap, f"gt_{plot_name}", save_path, tissue_class)

    # Remove unlabelled (class 0) ground truth points
    if remove_unlabelled:
        unlabelled_inds = tissue_class.nonzero()
        tissue_class = tissue_class[unlabelled_inds]
        predicted_labels = predicted_labels[unlabelled_inds]
        pos = pos[unlabelled_inds]

    # Evaluate against ground truth tissue annotations
    if run_id == 16:
        evaluate(tissue_class, predicted_labels)

    # Visualise cluster labels on graph patch
    print("Generating image")
    visualize_points(
        organ,
        save_path / f"patch_{plot_name}.png",
        pos,
        colours=predicted_labels,
        width=width,
        height=height,
    )


def evaluate(tissue_class, predicted_labels):
    accuracy = accuracy_score(tissue_class, predicted_labels)
    f1_micro = f1_score(tissue_class, predicted_labels, average="micro")
    try:
        top_3_accuracy = top_k_accuracy_score(tissue_class, predicted_labels, k=3)
    except ValueError:
        top_3_accuracy = 0
    print("-----------------------")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Top 3 accuracy: {top_3_accuracy:.3f}")
    print(f"F1 micro score: {f1_micro:.3f}")
    print("-----------------------")


if __name__ == "__main__":
    typer.run(main)
