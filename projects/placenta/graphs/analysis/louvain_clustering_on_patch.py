from pathlib import Path

import typer
from networkx.readwrite.gml import read_gml
import matplotlib.pyplot as plt
from torch_geometric.utils.convert import from_networkx
import numpy as np

from happy.utils.utils import get_project_dir
import happy.db.eval_runs_interface as db


def main(run_id: int = -1, width: int = -1, height: int = -1, name: str = ""):
    """Saves a visualisation of the result of Louvain clustering on graph

    Args:
        run_id: id of the run which generated the embeddings file
        width: width of the patch of the graph
        height: height of the patch of the graph
        name: name of the graph file

    """
    db.init()
    eval_run = db.get_eval_run_by_id(run_id)

    project_dir = get_project_dir('placenta')
    graph_dir = (
        project_dir
        / "datasets"
        / "graph"
        / f"lab_{eval_run.slide.lab.id}"
        / f"h{height}_w{width}"
    )
    graph_dir.mkdir(parents=True, exist_ok=True)
    graph_path = graph_dir / f"{name}.gml"

    # Make graph data object
    print(f"Reading graph object from gml file at {graph_path}")
    graph = read_gml(graph_path)
    data = from_networkx(graph)
    data.pos = np.array(list(zip(data.posx, data.posy)))
    data.LouvainCluster = [int(x.split(" ")[-1]) for x in data.LouvainCluster]

    print("Plotting graph...")
    vis_cluster(data, width, height, eval_run, name, project_dir)


def vis_cluster(data, width, height, eval_run, name, project_dir):
    slide_id = eval_run.slide.lab.id
    slide_name = eval_run.slide.slide_name
    save_dir = Path(f"lab_{slide_id}") / f"slide_{slide_name}"
    save_path = (
        project_dir
        / "visualisations"
        / "graphs"
        / save_dir
        / f"w{width}_h{height}"
        / "clustering"
        / "louvain"
    )
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"Plotting graph with clusters...")
    save_file = f"{name.split('_')[-1]}.png"
    visualize_points(
        save_path / save_file,
        data.pos,
        labels=data.LouvainCluster,
        edge_index=data.edge_index,
    )
    print(f"Plot saved to {save_path / save_file}")


def visualize_points(save_path, pos, labels=None, edge_index=None):
    fig = plt.figure(figsize=(8, 8), dpi=150)
    if edge_index is not None:
        for edge in edge_index.tolist():
            src = edge[0]
            dst = edge[1]
            src = pos[src].tolist()
            dst = pos[dst].tolist()
            plt.plot([src[0], dst[0]], [src[1], dst[1]], linewidth=0.5, color="black")
    plt.scatter(pos[:, 0], pos[:, 1], s=1, zorder=1000, cmap="tab20c", c=labels)
    plt.gca().invert_yaxis()
    plt.axis("off")
    fig.tight_layout()
    plt.savefig(save_path)


if __name__ == "__main__":
    typer.run(main)
