from pathlib import Path
import time

import typer
import torch
from torch_geometric.transforms import SIGN
from torch_geometric.loader import (
    NeighborSampler,
    NeighborLoader,
    ShaDowKHopSampler,
    DataLoader,
)
from scipy.special import softmax
import pandas as pd
import numpy as np

import happy.db.eval_runs_interface as db
from happy.utils.utils import get_device, get_project_dir
from happy.organs import get_organ
from happy.graph.graph_creation.get_and_process import get_hdf5_data
from projects.placenta.graphs.processing.process_knots import process_knts
from happy.utils.utils import set_seed
from happy.graph.enums import FeatureArg, MethodArg
from happy.graph.utils.visualise_points import visualize_points
from graphs.graphs.graph_supervised import inference, inference_mlp
from happy.graph.graph_creation.node_dataset_splits import setup_node_splits
from happy.graph.graph_creation.create_graph import setup_graph
from happy.hdf5 import get_embeddings_file


def main(
    seed: int = 0,
    project_name: str = "placenta",
    organ_name: str = "placenta",
    exp_name: str = typer.Option(...),
    model_weights_dir: str = typer.Option(...),
    model_name: str = typer.Option(...),
    run_id: int = typer.Option(...),
    k: int = 8,
    feature: FeatureArg = FeatureArg.embeddings,
    model_type: str = "sup_graphsage",
    graph_method: MethodArg = MethodArg.k,
    save_tsv: bool = False,
    save_embeddings: bool = False,
):
    db.init()
    set_seed(seed)
    device = get_device()
    project_dir = get_project_dir(project_name)
    organ = get_organ(organ_name)
    patch_files = [project_dir / "graph_splits" / file for file in ["all_wsi.csv"]]

    print("Begin graph construction...")
    hdf5_data = get_hdf5_data(project_name, run_id, 0, 0, -1, -1)
    # Covert isolated knts into syn and turn groups into a single knt point
    hdf5_data, _ = process_knts(organ, hdf5_data)
    # Covert input cell data into a graph
    data = setup_graph(hdf5_data, organ, feature, k, graph_method)
    # Split graph into an inference set across the WSI
    data = setup_node_splits(data, None, False, True, patch_files)
    x = data.x.to(device)
    print("Graph construction complete")

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
    if model_type == "sup_sign":
        data = SIGN(model.num_layers)(data)  # precompute SIGN fixed embeddings

    # Setup paths
    save_path = (
        Path(*pretrained_path.parts[:-1]) / "eval" / model_epochs / f"run_{run_id}"
    )
    save_path.mkdir(parents=True, exist_ok=True)
    plot_name = "all_wsi"

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
    elif model_type == "sup_shadow":
        eval_loader = ShaDowKHopSampler(
            data,
            depth=6,
            num_neighbors=5,
            node_idx=None,
            batch_size=4000,
            shuffle=False,
        )
    elif model_type == "sup_sign" or model_type == "sup_mlp":
        eval_loader = DataLoader(range(data.num_nodes), batch_size=512, shuffle=False)
    else:
        eval_loader = NeighborSampler(
            data.edge_index,
            node_idx=None,
            sizes=[-1],
            batch_size=512,
            shuffle=False,
        )

    # Run inference and get predicted labels for nodes
    timer_start = time.time()
    if model_type == "sup_mlp" or model_type == "sup_sign":
        out, graph_embeddings, predicted_labels = inference_mlp(
            model, data, eval_loader, device
        )
    else:
        out, graph_embeddings, predicted_labels = inference(
            model, data, eval_loader, device
        )
    timer_end = time.time()
    print(f"total time: {timer_end - timer_start:.4f} s")

    # Print some prediction count info
    tissue_label_mapping = {tissue.id: tissue.label for tissue in organ.tissues}
    _print_prediction_stats(predicted_labels, tissue_label_mapping)

    # Visualise cluster labels on graph patch
    print("Generating image")
    colours_dict = {tissue.id: tissue.colour for tissue in organ.tissues}
    colours = [colours_dict[label] for label in predicted_labels]
    visualize_points(
        organ,
        save_path / f"{plot_name.split('.png')[0]}.png",
        data.pos,
        colours=colours,
        width=int(data.pos[:, 0].max()) - int(data.pos[:, 0].min()),
        height=int(data.pos[:, 1].max()) - int(data.pos[:, 1].min()),
    )

    # Convert outputs to confidence scores
    tissue_confidence = softmax(out, axis=-1)
    top_confidence = tissue_confidence[
        (range(len(predicted_labels)), [predicted_labels])
    ][0]

    # Save predictions to tsv for loading into QuPath
    if save_tsv:
        _save_tissue_preds_as_tsv(predicted_labels, hdf5_data.coords, save_path, organ)

    # Save processed cell and tissue predictions, coordinates and embeddings
    embeddings_path = get_embeddings_file(project_name, run_id, tissue=True)
    if save_embeddings:
        _save_embeddings_as_hdf5(
            hdf5_data,
            predicted_labels,
            graph_embeddings,
            top_confidence,
            embeddings_path,
        )


def _print_prediction_stats(predicted_labels, tissue_label_mapping):
    unique, counts = np.unique(predicted_labels, return_counts=True)
    unique_labels = []
    for label in unique:
        unique_labels.append(tissue_label_mapping[label])
    unique_counts = dict(zip(unique_labels, counts))
    print(f"Counts per label: {unique_counts}")


def _save_embeddings_as_hdf5(
    hdf5_data, tissue_predictions, tissue_embeddings, tissue_confidence, save_path
):
    print("Saving all tissue predictions and embeddings to hdf5")
    hdf5_data.tissue_predictions = tissue_predictions
    hdf5_data.tissue_embeddings = tissue_embeddings
    hdf5_data.tissue_confidence = tissue_confidence
    hdf5_data.to_path(save_path)


def _save_tissue_preds_as_tsv(predicted_labels, coords, save_path, organ):
    print("Saving all tissue predictions as a tsv")
    label_dict = {tissue.id: tissue.label for tissue in organ.tissues}
    predicted_labels = [label_dict[label] for label in predicted_labels]
    tissue_preds_df = pd.DataFrame(
        {
            "x": coords[:, 0].astype(int),
            "y": coords[:, 1].astype(int),
            "class": predicted_labels,
        }
    )
    tissue_preds_df.to_csv(save_path / "tissue_preds.tsv", sep="\t", index=False)


if __name__ == "__main__":
    typer.run(main)
