from pathlib import Path

import typer
import torch
from sklearn.metrics import (
    accuracy_score,
    top_k_accuracy_score,
    f1_score,
    confusion_matrix,
    cohen_kappa_score,
    roc_auc_score,
    matthews_corrcoef
)
from sklearn.metrics import precision_recall_fscore_support as score
import numpy as np
import pandas as pd
from torch_geometric.data import NeighborSampler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import softmax

from happy.utils.utils import get_device, get_project_dir
from happy.train.utils import plot_confusion_matrix
from happy.organs.organs import get_organ
from graphs.graphs.create_graph import get_raw_data, setup_graph
from graphs.graphs.embeddings import fit_umap, plot_cell_graph_umap, plot_tissue_umap
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
    label_type: str = "full",
    tissue_label_tsv: str = "139_tissue_points.tsv",
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
    _, _, tissue_class = get_groundtruth_patch(
        organ, project_dir, x_min, y_min, width, height, tissue_label_tsv, label_type
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
    out, graph_embeddings, predicted_labels = _get_model_predictions(
        model, x, eval_loader, device
    )

    # Remove unlabelled (class 0) ground truth points
    if remove_unlabelled and run_id == 16:
        unlabelled_inds, tissue_class, predicted_labels, pos, out = _remove_unlabelled(
            tissue_class, predicted_labels, pos, out
        )

        if plot_umap:
            graph_embeddings = graph_embeddings[unlabelled_inds]
            predictions = predictions[unlabelled_inds]
            # fit and plot umap with cell classes
            fitted_umap = fit_umap(graph_embeddings)
            plot_cell_graph_umap(
                organ, predictions, fitted_umap, save_path, f"eval_{plot_name}.png"
            )
            # Plot the predicted labels onto the umap of the graph embeddings
            plot_tissue_umap(organ, fitted_umap, plot_name, save_path, predicted_labels)
            if run_id == 16:
                plot_tissue_umap(
                    organ, fitted_umap, f"gt_{plot_name}", save_path, tissue_class
                )

    # Print some prediction count info
    _print_prediction_stats(predicted_labels)

    # Evaluate against ground truth tissue annotations
    if run_id == 16:
        evaluate(
            tissue_class,
            predicted_labels,
            out,
            organ,
            save_path,
            remove_unlabelled,
            label_type,
        )

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


def evaluate(
    tissue_class, predicted_labels, out, organ, run_path, remove_unlabelled, label_type
):
    accuracy = accuracy_score(tissue_class, predicted_labels)
    f1_macro = f1_score(tissue_class, predicted_labels, average="macro")
    top_3_accuracy = top_k_accuracy_score(tissue_class, out, k=3)
    cohen_kappa = cohen_kappa_score(tissue_class, predicted_labels)
    mcc = matthews_corrcoef(tissue_class, predicted_labels)
    roc_auc = roc_auc_score(
        tissue_class, softmax(out, axis=-1), average="macro", multi_class="ovr"
    )
    weighted_roc_auc = roc_auc_score(
        tissue_class, softmax(out, axis=-1), average="weighted", multi_class="ovr"
    )

    all_scores = score(tissue_class, predicted_labels)

    print("-----------------------")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Top 3 accuracy: {top_3_accuracy:.3f}")
    print(f"F1 macro score: {f1_macro:.3f}")
    print(f"Cohen's Kappa score: {cohen_kappa:.3f}")
    print(f"MCC score: {mcc:.3f}")
    print(f"ROC AUC macro: {roc_auc:.3f}")
    print(f"Weighted ROC AUC macro: {weighted_roc_auc:.3f}")
    print("-----------------------")
    print([tissue.label for tissue in organ.tissues][-9:])
    print(f'precision: {all_scores[0].round(3)}')
    print(f'recall: {all_scores[1].round(3)}')
    print(f'fscore: {all_scores[2].round(3)}')
    print(f'support: {all_scores[3].round(3)}')
    print("-----------------------")

    if label_type == "full":
        cell_labels = [tissue.label for tissue in organ.tissues]
    elif label_type == "alt":
        cell_labels = np.array([tissue.alt_label for tissue in organ.tissues])[
            [0, 1, 3, 4, 7, 8, 9]
        ]
    elif label_type == "tissue":
        cell_labels = np.array([tissue.alt_label for tissue in organ.tissues])[
            [0, 1, 7, 8]
        ]
    else:
        raise ValueError(f"No such label type: {label_type}")

    cm = confusion_matrix(predicted_labels, tissue_class)
    if remove_unlabelled:
        cell_labels = cell_labels[-len(cell_labels) + 1 :]
        cm = cm[-9:, -9:] # Might have to change this for the other labels
    cm_df = pd.DataFrame(cm, columns=cell_labels, index=cell_labels).astype(int)

    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1.1)
    plot_confusion_matrix(cm_df / len(tissue_class), "9 Tissue", run_path, ".1%")


def _remove_unlabelled(tissue_class, predicted_labels, pos, out):
    labelled_inds = tissue_class.nonzero()
    tissue_class = tissue_class[labelled_inds]
    pos = pos[labelled_inds]
    out = out[labelled_inds]
    out = np.delete(out, 0, axis=1)
    predicted_labels = predicted_labels[labelled_inds]
    return labelled_inds, tissue_class, predicted_labels, pos, out


@torch.no_grad()
def _get_model_predictions(model, x, eval_loader, device):
    print("Running inference")
    model.eval()
    out, graph_embeddings = model.inference(x, eval_loader, device)
    predicted_labels = out.argmax(dim=-1, keepdim=True).squeeze()
    predicted_labels = predicted_labels.cpu().numpy()
    out = out.cpu().detach().numpy()
    return out, graph_embeddings, predicted_labels


def _print_prediction_stats(predicted_labels):
    unique, counts = np.unique(predicted_labels, return_counts=True)
    unique_counts = dict(zip(unique, counts))
    print(f"Predictions per label: {unique_counts}")


if __name__ == "__main__":
    typer.run(main)
