import time
import os
import warnings
import copy

import numpy as np
import typer
import torch

import happy.db.eval_runs_interface as db
from happy.utils.utils import get_device, get_project_dir
from happy.organs import get_organ
from happy.utils.utils import set_seed
from happy.graph.enums import AutoEncoderModelsArg
from happy.graph.utils.visualise_points import visualize_points
from happy.graph.utils.utils import get_model_eval_path
from happy.graph.graph_creation.get_and_process import get_hdf5_data
from happy.models.utils.custom_layers import KnnEdges


def main(
    seed: int = 0,
    project_name: str = "placenta",
    organ_name: str = "placenta",
    exp_name: str = typer.Option(...),
    model_weights_dir: str = typer.Option(...),
    model_name: str = typer.Option(...),
    run_id: int = typer.Option(...),
    model_type: AutoEncoderModelsArg = AutoEncoderModelsArg.one_hop,
    subsample_ratio: float = 0.25,
    plot_edges: bool = True,
    plot_scores: bool = True,
):
    db.init()
    set_seed(seed)
    device = get_device()
    project_dir = get_project_dir(project_name)
    organ = get_organ(organ_name)

    # Get hdf5 for the original cell and tissue labels
    if not plot_scores:
        hdf5_data = get_hdf5_data(project_name, run_id, 0, 0, -1, -1, tissue=True)
        cell_predictions = hdf5_data.cell_predictions
        tissue_predictions = hdf5_data.tissue_predictions

    # Get graph to infer over
    data_dir = project_dir / "datasets" / "lesion"
    data_file_name = f"run_{run_id}.pt"
    if os.path.exists(data_dir / "single" / data_file_name):
        data = torch.load(data_dir / "single" / data_file_name)
    elif os.path.exists(data_dir / "multi" / data_file_name):
        data = torch.load(data_dir / "multi" / data_file_name)
    else:
        raise ValueError(f"Could not find data file {data_file_name}")
    data.edge_weights = data.edge_attr[:, 0]
    data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
    if subsample_ratio > 0.0:
        num_to_keep = int(data.num_nodes * subsample_ratio)
        keep_indices = np.random.choice(
            np.arange(data.num_nodes), num_to_keep, replace=False
        )
        data = data.subgraph(torch.LongTensor(keep_indices))
    data = data.to(device)

    # Setup trained model
    pretrained_path = (
        project_dir
        / "results"
        / "gae"
        / model_type.value
        / exp_name
        / model_weights_dir
        / model_name
    )
    model = torch.load(pretrained_path, map_location=device)

    timer_start = time.time()
    # Run inference and get predicted labels for nodes
    print(f"Running inference on run {run_id} with forward hooks to get pooled graphs")
    pooling_outputs = get_pooled_graph(model, data)
    timer_end = time.time()
    print(f"total inference time: {timer_end - timer_start:.4f} s")

    # Get data back onto cpu for results saving
    data = data.to("cpu")
    edge_index = data.edge_index

    # Setup path to save results
    save_path = get_model_eval_path(model_name, pretrained_path, run_id)
    # Visualise original graph
    print(f"Generating image for original graph")
    if plot_scores:
        plot_name = f"original.png"
        colours = ["black" for _ in range(len(data.pos))]
    else:
        plot_name = f"original_cells.png"
        colours_dict = {cell.id: cell.colour for cell in organ.cells}
        colours = [colours_dict[label] for label in hdf5_data.cell_predictions]
        visualize_points(
            organ,
            save_path / plot_name,
            data.pos,
            colours=colours,
            width=int(data.pos[:, 0].max()) - int(data.pos[:, 0].min()),
            height=int(data.pos[:, 1].max()) - int(data.pos[:, 1].min()),
            edge_index=edge_index,
            point_size=0.05,
            )
        plot_name = f"original_tissues.png"
        colours_dict = {tissue.id: tissue.colour for tissue in organ.tissues}
        colours = [colours_dict[label] for label in hdf5_data.tissue_predictions]
    visualize_points(
        organ,
        save_path / plot_name,
        data.pos,
        colours=colours,
        width=int(data.pos[:, 0].max()) - int(data.pos[:, 0].min()),
        height=int(data.pos[:, 1].max()) - int(data.pos[:, 1].min()),
        edge_index=data.edge_index,
        point_size=0.05,
        )

    # Visualise pooled graphs
    for i, pooled_output in enumerate(pooling_outputs):
        pos = pooled_output[1].to("cpu")
        edge_index = pooled_output[2].to("cpu")
        perm = pooled_output[5].to("cpu")
        scores = pooled_output[6]
        if scores is None:
            scores = ["black" for _ in range(len(pos))]
        else:
            scores = scores.numpy()
        if not plot_edges:
            edge_index = None

        print(f"Generating image for pooling layer {i}")
        if plot_scores:
            plot_name = f"pool_{i}.png"
            colours = scores
        else:
            plot_name = f"pool_{i}_cells.png"
            cell_predictions = cell_predictions[perm]
            colours_dict = {cell.id: cell.colour for cell in organ.cells}
            colours = np.array([colours_dict[label] for label in cell_predictions])
            visualize_points(
                organ,
                save_path / plot_name,
                pos,
                colours=colours,
                width=int(pos[:, 0].max()) - int(pos[:, 0].min()),
                height=int(pos[:, 1].max()) - int(pos[:, 1].min()),
                edge_index=edge_index,
                point_size=0.05,
                )
            plot_name = f"pool_{i}_tissues.png"
            tissue_predictions = tissue_predictions[perm]
            colours_dict = {tissue.id: tissue.colour for tissue in organ.tissues}
            colours = np.array([colours_dict[label] for label in tissue_predictions])
        visualize_points(
            organ,
            save_path / plot_name,
            pos,
            colours=colours,
            width=int(pos[:, 0].max()) - int(pos[:, 0].min()),
            height=int(pos[:, 1].max()) - int(pos[:, 1].min()),
            edge_index=edge_index,
            point_size=0.05,
            )


@torch.no_grad()
def get_pooled_graph(model, data):
    pooling_outputs = []

    def hook(model, input, output):
        # Clone output in case it will be later modified in-place:
        pooling_outputs.append(copy.copy(output))

    hook_handles = []
    for module in model.modules():
        if isinstance(module, KnnEdges):
            hook_handles.append(module.register_forward_hook(hook))

    if len(hook_handles) == 0:
        warnings.warn("The 'model' does not have any 'KnnEdges' layers")

    model.eval()
    with torch.no_grad():
        model(data)

    for handle in hook_handles:
        handle.remove()

    return pooling_outputs


if __name__ == "__main__":
    typer.run(main)
