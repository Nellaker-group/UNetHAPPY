from pathlib import Path
from typing import Optional

import typer
import torch
from sklearn.metrics import (
    accuracy_score,
    top_k_accuracy_score,
    f1_score,
    cohen_kappa_score,
    roc_auc_score,
    matthews_corrcoef,
)
from torch_geometric.transforms import ToUndirected
from torch_geometric.utils import add_self_loops
from torch_geometric.loader import NeighborSampler, NeighborLoader
from scipy.special import softmax
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from happy.utils.utils import get_device, get_project_dir
from happy.train.utils import (
    plot_confusion_matrix,
    plot_tissue_pr_curves,
    get_tissue_confusion_matrix,
)
from happy.organs.organs import get_organ
from graphs.graphs.create_graph import get_raw_data, setup_graph
from graphs.graphs.embeddings import fit_umap, plot_cell_graph_umap, plot_tissue_umap
from graphs.graphs.utils import get_feature
from graphs.graphs.enums import FeatureArg, MethodArg
from graphs.analysis.vis_graph_patch import visualize_points
from graphs.graphs.create_graph import get_groundtruth_patch
from graphs.graphs.graph_supervised import inference

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
    tissue_label_tsv: Optional[str] = None,
):
    device = get_device()
    project_dir = get_project_dir(project_name)
    organ = get_organ(organ_name)

    # Get data from hdf5 files
    predictions, embeddings, coords, confidence = get_raw_data(
        project_name, run_id, x_min, y_min, width, height, top_conf
    )

    feature_data = get_feature(feature.value, predictions, embeddings)
    data = setup_graph(coords, k, feature_data, graph_method.value, loop=False)
    data = ToUndirected()(data)
    data.edge_index, data.edge_attr = add_self_loops(
        data["edge_index"], data["edge_attr"], fill_value="mean"
    )
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
    if model_type == "sup_graphsage":
        eval_loader = NeighborLoader(
            data,
            num_neighbors=[-1],
            batch_size=512,
            shuffle=False,
        )
        eval_loader.data.num_nodes = data.num_nodes
        eval_loader.data.n_id = torch.arange(data.num_nodes)
    else:
        eval_loader = NeighborSampler(
            data.edge_index,
            node_idx=None,
            sizes=[-1],
            batch_size=512,
            shuffle=False,
        )

    # Run inference and get predicted labels for nodes
    out, graph_embeddings, predicted_labels = inference(model, x, eval_loader, device)

    # Remove unlabelled (class 0) ground truth points
    if remove_unlabelled and tissue_label_tsv is not None:
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
            if tissue_label_tsv is not None:
                plot_tissue_umap(
                    organ, fitted_umap, f"gt_{plot_name}", save_path, tissue_class
                )

    # Print some prediction count info
    tissue_label_mapping = {tissue.id: tissue.label for tissue in organ.tissues}
    _print_prediction_stats(predicted_labels, tissue_label_mapping)

    # Evaluate against ground truth tissue annotations
    if tissue_label_tsv is not None:
        evaluate(
            tissue_class,
            predicted_labels,
            out,
            organ,
            save_path,
            remove_unlabelled,
        )

    # Visualise cluster labels on graph patch
    print("Generating image")
    colours_dict = {tissue.id: tissue.colour for tissue in organ.tissues}
    colours = [colours_dict[label] for label in predicted_labels]
    visualize_points(
        organ,
        save_path / f"patch_{plot_name}.png",
        pos,
        colours=colours,
        width=width,
        height=height,
    )

    if x_min is None:
        label_dict = {tissue.id: tissue.label for tissue in organ.tissues}
        predicted_labels = [label_dict[label] for label in predicted_labels]
        _save_tissue_preds_as_tsv(predicted_labels, coords, save_path)


def evaluate(tissue_class, predicted_labels, out, organ, run_path, remove_unlabelled):
    tissue_ids = [tissue.id for tissue in organ.tissues]
    if remove_unlabelled:
        tissue_ids = tissue_ids[1:]

    accuracy = accuracy_score(tissue_class, predicted_labels)
    f1_macro = f1_score(tissue_class, predicted_labels, average="macro")
    top_3_accuracy = top_k_accuracy_score(tissue_class, out, k=3, labels=tissue_ids)
    cohen_kappa = cohen_kappa_score(tissue_class, predicted_labels)
    mcc = matthews_corrcoef(tissue_class, predicted_labels)
    roc_auc = roc_auc_score(
        tissue_class,
        softmax(out, axis=-1),
        average="macro",
        multi_class="ovo",
        labels=tissue_ids,
    )
    weighted_roc_auc = roc_auc_score(
        tissue_class,
        softmax(out, axis=-1),
        average="weighted",
        multi_class="ovo",
        labels=tissue_ids,
    )
    print("-----------------------")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Top 3 accuracy: {top_3_accuracy:.3f}")
    print(f"F1 macro score: {f1_macro:.3f}")
    print(f"Cohen's Kappa score: {cohen_kappa:.3f}")
    print(f"MCC score: {mcc:.3f}")
    print(f"ROC AUC macro: {roc_auc:.3f}")
    print(f"Weighted ROC AUC macro: {weighted_roc_auc:.3f}")
    print("-----------------------")

    cm_df, cm_df_props = get_tissue_confusion_matrix(
        organ, predicted_labels, tissue_class, proportion_label=True
    )
    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1.1)
    plot_confusion_matrix(cm_df, "All Tissues", run_path, "d")
    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1.1)
    plot_confusion_matrix(cm_df_props, "All Tissues Proportion", run_path, ".2f")

    tissue_mapping = {tissue.id: tissue.label for tissue in organ.tissues}
    tissue_colours = {tissue.id: tissue.colour for tissue in organ.tissues}
    plot_tissue_pr_curves(
        tissue_mapping,
        tissue_colours,
        tissue_class,
        predicted_labels,
        out,
        run_path / "pr_curves.png",
    )


def _remove_unlabelled(tissue_class, predicted_labels, pos, out):
    labelled_inds = tissue_class.nonzero()
    tissue_class = tissue_class[labelled_inds]
    pos = pos[labelled_inds]
    out = out[labelled_inds]
    out = np.delete(out, 0, axis=1)
    predicted_labels = predicted_labels[labelled_inds]
    return labelled_inds, tissue_class, predicted_labels, pos, out


def _print_prediction_stats(predicted_labels, tissue_label_mapping):
    unique, counts = np.unique(predicted_labels, return_counts=True)
    unique_labels = []
    for label in unique:
        unique_labels.append(tissue_label_mapping[label])
    unique_counts = dict(zip(unique_labels, counts))
    print(f"Predictions per label: {unique_counts}")


def _save_tissue_preds_as_tsv(predicted_labels, coords, save_path):
    tissue_preds_df = pd.DataFrame(
        {"x": coords[:, 0], "y": coords[:, 1], "class": predicted_labels}
    )
    tissue_preds_df.to_csv(save_path / "tissue_preds.tsv", sep="\t", index=False)


if __name__ == "__main__":
    typer.run(main)
