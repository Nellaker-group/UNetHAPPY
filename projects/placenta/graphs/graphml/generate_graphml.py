import typer
from networkx.readwrite.graphml import write_graphml
import torch
from torch_cluster import knn_graph
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx

import happy.db.eval_runs_interface as db
from happy.utils.hdf5 import get_datasets_in_patch, get_embeddings_file
from happy.utils.utils import get_project_dir


def main(
    run_id: int = typer.Option(...),
    x_min: int = 0,
    y_min: int = 0,
    width: int = -1,
    height: int = -1,
    k: int = 5,
):
    """Create a graph and save it as a .graphml file

    Args:
        run_id: id of the run which generated the embeddings file
        x_min: min x coordinate for defining a subsection/patch of the WSI
        y_min: min y coordinate for defining a subsection/patch of the WSI
        width: width for defining a subsection/patch of the WSI. -1 for all
        height: height for defining a subsection/patch of the WSI. -1 for all
        k: nearest neighbours for creating the graph edges
    """
    patch = (
        False if x_min == 0 and y_min == 0 and width == -1 and height == -1 else True
    )

    project_dir = get_project_dir("placenta")

    # Create database connection
    db.init()
    eval_run = db.get_eval_run_by_id(run_id)

    # Get path to embeddings hdf5 files
    embeddings_path = get_embeddings_file("placenta", run_id)
    print(f"Getting data from: {embeddings_path}")

    # Get hdf5 datasets contained in specified box/patch of WSI
    predictions, embeddings, coords, confidence = get_datasets_in_patch(
        embeddings_path, x_min, y_min, width, height
    )
    print(f"Data loaded with {len(predictions)} nodes")

    # Make graph data object
    print("Generating graph")
    data = Data(y=torch.Tensor(predictions), pos=torch.Tensor(coords.astype("int32")))
    data.edge_index = knn_graph(data.pos, k=k)
    data.pos_x = data.pos[:, 0]
    data.pos_y = data.pos[:, 1]
    graph = to_networkx(data, to_undirected=True, node_attrs=["pos_x", "pos_y", "y"])

    save_dir = project_dir / "datasets" / "graph" / f"lab_{eval_run.slide.lab.id}"
    save_dir = save_dir / f"h{height}_w{width}" if patch else save_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"{eval_run.slide.slide_name}.graphml"
    save_path = save_dir / file_name

    print(f"Saving graphml file to {save_path / file_name}")
    write_graphml(graph, save_path / file_name)


if __name__ == "__main__":
    typer.run(main)
