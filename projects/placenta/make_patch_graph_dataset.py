import typer
import torch
import numpy as np

from graphs.graphs.create_graph import get_raw_data, setup_graph
from happy.organs.organs import get_organ
from happy.utils.utils import get_project_dir
from graphs.graphs.enums import MethodArg


def main(
    project_name: str = "placenta",
    organ_name: str = "placenta",
    run_id: int = typer.Option(...),
    x_min: int = 0,
    y_min: int = 0,
    width: int = -1,
    height: int = -1,
    k: int = 6,
    graph_method: MethodArg = MethodArg.k,
):
    project_dir = get_project_dir(project_name)
    organ = get_organ(organ_name)

    # Get raw data from hdf5 across WSI
    predictions, _, coords, confidence = get_raw_data(
        project_name, run_id, x_min, y_min, width, height
    )

    # Create the graph from the raw data
    data = setup_graph(coords, k, predictions, graph_method.value)

    # Remove points from graph with edges which are too long (sort by edge length)
    edge_weight_limit = 0.95
    edge_lengths = data["edge_attr"].numpy().ravel()
    sorted_inds_over_length = (np.sort(edge_lengths) > edge_weight_limit).nonzero()[0]
    edge_index_inds = np.argsort(data["edge_attr"].numpy().ravel())[
        sorted_inds_over_length
    ]
    data['x'] = _remove_element_by_indicies(data['x'], edge_index_inds)
    data['pos'] = _remove_element_by_indicies(data['pos'], edge_index_inds)
    data['edge_index'] = _remove_element_by_indicies(data['edge_index'], edge_index_inds)
    data['edge_attr'] = _remove_element_by_indicies(data['edge_attr'], edge_index_inds)
    print(f"Removed {len(edge_index_inds)} points from graph")

    # Make list of tile x,y coordinates with width and height

    # Remove tiles without any cell points

    # For each tile, generate the tile specific graph

    # Quantify cell types within tiles

    # Quantify cell connections within tiles


def _remove_element_by_indicies(tensor, inds):
    if len(tensor) > 2:
        mask = torch.ones(tensor.size()[0], dtype=torch.bool)
        mask[inds] = False
        return tensor[mask]
    else:
        mask = torch.ones(tensor.size()[1], dtype=torch.bool)
        mask[inds] = False
        return tensor[:, mask]


if __name__ == "__main__":
    typer.run(main)
