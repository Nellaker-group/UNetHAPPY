from pathlib import Path
import time
import os

import typer
import torch
from torch_geometric.transforms import ToUndirected, SIGN
from torch_geometric.utils import add_self_loops
from torch_geometric.loader import (
    NeighborSampler,
    NeighborLoader,
    ShaDowKHopSampler,
    DataLoader,
)
from scipy.special import softmax
import numpy as np
import h5py

import happy.db.eval_runs_interface as db
from happy.utils.utils import get_device, get_project_dir
from happy.organs import get_organ
from happy.graph.create_graph import get_raw_data, setup_graph, process_knts
from graphs.graphs.utils import get_feature
from happy.utils.utils import set_seed
from graphs.graphs.enums import FeatureArg, MethodArg
from graphs.analysis.vis_graph_patch import visualize_points
from graphs.graphs.graph_supervised import inference, setup_node_splits, inference_mlp
from happy.utils.hdf5 import get_embeddings_file


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
    save_embeddings: bool = False,
):
    db.init()
    set_seed(seed)
    device = get_device()
    project_dir = get_project_dir(project_name)
    organ = get_organ(organ_name)
    patch_files = [project_dir / "graph_splits" / file for file in ["all_wsi.csv"]]

    print("Begin graph construction...")
    predictions, embeddings, coords, confidence = get_raw_data(
        project_name, run_id, 0, 0, -1, -1
    )
    # Covert isolated knts into syn and turn groups into a single knt point
    predictions, embeddings, coords, confidence, _ = process_knts(
        organ, predictions, embeddings, coords, confidence
    )
    # Covert input cell data into a graph
    feature_data = get_feature(feature, predictions, embeddings, organ)
    data = setup_graph(coords, k, feature_data, graph_method, loop=False)
    data = ToUndirected()(data)
    data.edge_index, data.edge_attr = add_self_loops(
        data["edge_index"], data["edge_attr"], fill_value="mean"
    )
    pos = data.pos
    x = data.x.to(device)

    data = setup_node_splits(data, None, False, True, patch_files)
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
            model, x, eval_loader, device
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
        pos,
        colours=colours,
        width=int(data.pos[:, 0].max()) - int(data.pos[:, 0].min()),
        height=int(data.pos[:, 1].max()) - int(data.pos[:, 1].min()),
    )

    # Convert outputs to confidence scores
    tissue_confidence = softmax(out, axis=-1)
    top_confidence = tissue_confidence[
        (range(len(predicted_labels)), [predicted_labels])
    ][0]

    embeddings_path = str(get_embeddings_file(project_name, run_id)).split(".hdf5")[0]
    # Save processed cell and tissue predictions, coordinates and embeddings
    if save_embeddings:
        _save_embeddings_as_hdf5(
            predictions,
            embeddings,
            confidence,
            coords,
            predicted_labels,
            graph_embeddings,
            top_confidence,
            f"{embeddings_path}_tissues.hdf5",
        )


def _print_prediction_stats(predicted_labels, tissue_label_mapping):
    unique, counts = np.unique(predicted_labels, return_counts=True)
    unique_labels = []
    for label in unique:
        unique_labels.append(tissue_label_mapping[label])
    unique_counts = dict(zip(unique_labels, counts))
    print(f"Counts per label: {unique_counts}")


def _save_embeddings_as_hdf5(
    cell_predictions,
    cell_embeddings,
    cell_confidence,
    coords,
    tissue_predictions,
    tissue_embeddings,
    tissue_confidence,
    save_path,
):
    print("Saving all tissue predictions and embeddings to hdf5")

    if not os.path.isfile(save_path):
        total = len(cell_predictions)
        with h5py.File(save_path, "w-") as f:
            f.create_dataset(
                "cell_predictions", data=cell_predictions, shape=(total,), dtype="int8"
            )
            f.create_dataset(
                "cell_embeddings",
                data=cell_embeddings,
                shape=(total, 64),
                dtype="float32",
            )
            f.create_dataset(
                "cell_confidence", data=cell_confidence, shape=(total,), dtype="float16"
            )
            f.create_dataset("coords", data=coords, shape=(total, 2), dtype="uint32")
            f.create_dataset(
                "tissue_predictions",
                data=tissue_predictions,
                shape=(total,),
                dtype="int8",
            )
            f.create_dataset(
                "tissue_embeddings",
                data=tissue_embeddings,
                shape=(total, 64),
                dtype="float32",
            )
            f.create_dataset(
                "tissue_confidence",
                data=tissue_confidence,
                shape=(total,),
                dtype="float16",
            )


if __name__ == "__main__":
    typer.run(main)
