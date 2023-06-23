import os

import numpy as np
import typer
import torch
from torch_geometric.nn.pool import fps
from torch_geometric.data import Data

import happy.db.eval_runs_interface as db
from happy.utils.utils import get_device, get_project_dir
from happy.organs import get_organ
from happy.graph.utils.visualise_points import visualize_points
from happy.graph.graph_creation.get_and_process import get_hdf5_data
from happy.models.utils.custom_layers import KnnEdges


def main(
    project_name: str = "placenta",
    organ_name: str = "placenta",
    run_id: int = typer.Option(...),
    num_pooling_steps: int = typer.Option(...),
    subsample_ratio: float = 0.5,
    pooling_ratio: float = 0.5,
    plot_edges: bool = True,
):
    db.init()
    device = get_device()
    project_dir = get_project_dir(project_name)
    organ = get_organ(organ_name)

    # Get hdf5 for the original cell and tissue labels
    hdf5_data = get_hdf5_data(project_name, run_id, 0, 0, -1, -1, tissue=True)
    if subsample_ratio > 0.0:
        hdf5_data, mask = hdf5_data.filter_randomly(subsample_ratio)
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

    if subsample_ratio > 0.0:
        keep_indices = np.where(mask)[0]
        data = data.subgraph(torch.LongTensor(keep_indices))
    data = data.to(device)

    # Setup knn edge layer
    knn_edges = KnnEdges(start_k=6, k_increment=1, no_op=not plot_edges)

    # Apply FPS Pooling and edge reconstruction
    print(f"Applying fps pooling and knn edges if required")
    perms = []
    datas = []
    x = data.x
    pos = data.pos
    edge_index = data.edge_index
    batch = data.batch
    for i in range(num_pooling_steps):
        perm = fps(pos, ratio=pooling_ratio, batch=batch)
        perms.append(perm)
        x, pos, edge_index, _, batch, _, _ = knn_edges(
            x, pos, edge_index, None, batch, perm, None, i
        )
        datas.append(Data(x=x, pos=pos, edge_index=edge_index))
        print(f"Finished pooling step {i+1}")

    # Get data back onto cpu for results saving
    data = data.to("cpu")
    edge_index = data.edge_index
    save_path = (
        project_dir / "visualisations" / "graphs" / "fps_pooling" / f"run_{run_id}"
    )
    save_path.mkdir(parents=True, exist_ok=True)

    # Visualise original graph
    print(f"Generating image for original graphs")
    plot_name = f"original_cells.png"
    if not os.path.exists(save_path / plot_name):
        colours_dict = {cell.id: cell.colour for cell in organ.cells}
        colours = [colours_dict[label] for label in cell_predictions]
        visualize_points(
            organ,
            save_path / plot_name,
            data.pos,
            colours=colours,
            width=int(data.pos[:, 0].max()) - int(data.pos[:, 0].min()),
            height=int(data.pos[:, 1].max()) - int(data.pos[:, 1].min()),
            edge_index=edge_index,
        )
    plot_name = f"original_tissues.png"
    if not os.path.exists(save_path / plot_name):
        colours_dict = {tissue.id: tissue.colour for tissue in organ.tissues}
        colours = [colours_dict[label] for label in tissue_predictions]
        visualize_points(
            organ,
            save_path / plot_name,
            data.pos,
            colours=colours,
            width=int(data.pos[:, 0].max()) - int(data.pos[:, 0].min()),
            height=int(data.pos[:, 1].max()) - int(data.pos[:, 1].min()),
            edge_index=edge_index,
        )

    # Visualise pooled graphs
    for i, data in enumerate(datas):
        perm = perms[i].to("cpu")
        pos = data.pos.to("cpu")
        if plot_edges:
            edge_index = data.edge_index.to("cpu")
        else:
            pos = pos[perm]
            edge_index = None

        print(f"Generating image for pooling layer {i+1}")
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
        )


if __name__ == "__main__":
    typer.run(main)
