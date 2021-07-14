from pathlib import Path

import typer
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import networkx as nx
import torch
from torch_geometric.data import Data
from scipy.spatial import voronoi_plot_2d

import happy.db.eval_runs_interface as db
from happy.hdf5.utils import (
    get_datasets_in_patch,
    filter_by_confidence,
    get_embeddings_file,
)
from happy.utils.utils import get_project_dir
from happy.cells.cells import get_organ
from projects.placenta.graphs.graphs.create_graph import (
    make_k_graph,
    make_radius_k_graph,
    make_voronoi_graph,
    make_delaunay_graph,
)


def main(
    run_id: int = typer.Option(...),
    organ_name="placenta",
    method: str = "k",
    x_min: int = 0,
    y_min: int = 0,
    width: int = -1,
    height: int = -1,
    top_conf: bool = False,
):
    """Generates a graph and saves it's visualisation. Node are coloured by cell type

    Args:
        run_id: id of the run which generated the embeddings file
        organ_name: name of the organ to get the cell colours
        method: graph creation method to use. 'k', 'radius', 'voronoi' or 'delaunay'.
        x_min: min x coordinate for defining a subsection/patch of the WSI
        y_min: min y coordinate for defining a subsection/patch of the WSI
        width: width for defining a subsection/patch of the WSI. -1 for all
        height: height for defining a subsection/patch of the WSI. -1 for all
        top_conf: filter the nodes to only those >90% network confidence
    """
    # Create database connection
    db.init()

    organ = get_organ(organ_name)

    project_dir = get_project_dir(organ_name)

    # Get path to embeddings hdf5 files
    embeddings_path = get_embeddings_file(organ_name, run_id)
    print(f"Getting data from: {embeddings_path}")

    # Get hdf5 datasets contained in specified box/patch of WSI
    predictions, embeddings, coords, confidence = get_datasets_in_patch(
        embeddings_path, x_min, y_min, width, height
    )
    print(f"Data loaded with {len(predictions)} nodes")

    if top_conf:
        predictions, embeddings, coords, confidence = filter_by_confidence(
            predictions, embeddings, coords, confidence, 0.9, 1.0
        )

    # Make graph data object
    data = Data(x=predictions, pos=torch.Tensor(coords.astype("int32")))

    save_dirs = Path(*embeddings_path.parts[-3:-1])
    save_dir = (
        project_dir / "visualisations" / "graphs" / save_dirs / f"w{width}_h{height}"
    )
    plot_name = f"x{x_min}_y{y_min}_top_conf" if top_conf else f"x{x_min}_y{y_min}"

    if method == "k":
        vis_for_range_k(5, 6, data, plot_name, save_dir, organ)
    elif method == "radius":
        vis_for_range_radius(200, 260, 20, data, plot_name, save_dir, organ)
    elif method == "voronoi":
        vis_voronoi(data, plot_name, save_dir)
    elif method == "delaunay":
        vis_delaunay(data, plot_name, save_dir, organ)
    else:
        raise ValueError(f"no such method: {method}")


def vis_for_range_k(k_start, k_end, data, plot_name, save_dir, organ):
    # Specify save graph vis location
    save_path = save_dir / "max_radius"
    save_path.mkdir(parents=True, exist_ok=True)

    # Generate vis for different values of k
    for k in range(k_start, k_end):
        data = make_k_graph(data, k)

        plot_name = f"k{k}_{plot_name}.png"
        print(f"Plotting...")
        _visualize_points(
            organ,
            save_path / plot_name,
            data.pos,
            labels=data.x,
            edge_index=data.edge_index,
            edge_weight=data.edge_attr,
        )
        print(f"Plot saved to {save_path / plot_name}")


def vis_for_range_radius(rad_start, rad_end, k, data, plot_name, save_dir, organ):
    for radius in range(rad_start, rad_end, 10):
        # Specify save graph vis location
        save_path = save_dir / f"radius_{radius}"
        save_path.mkdir(parents=True, exist_ok=True)

        # Generate vis for radius and k
        data = make_radius_k_graph(data, radius, k)

        print(f"Plotting...")
        plot_name = f"k{k}_{plot_name}.png"
        _visualize_points(
            organ,
            save_path / plot_name,
            data.pos,
            labels=data.x,
            edge_index=data.edge_index,
        )
        print(f"Plot saved to {save_path / plot_name}")


def vis_voronoi(data, plot_name, save_dir):
    # Specify save graph vis location
    save_path = save_dir / "voronoi"
    save_path.mkdir(parents=True, exist_ok=True)

    vor = make_voronoi_graph(data)
    print(f"Plotting...")

    fig = voronoi_plot_2d(
        vor,
        show_vertices=False,
        line_colors="black",
        line_width=0.5,
        line_alpha=0.6,
        point_size=2,
        figsize=(8, 8),
        dpi=150,
    )
    plt.gca().invert_yaxis()
    plt.axis("off")
    fig.tight_layout()

    plot_name = f"{plot_name}.png"
    plt.savefig(save_path / plot_name)
    print(f"Plot saved to {save_path / plot_name}")


def vis_delaunay(data, plot_name, save_dir, organ):
    colours_dict = {cell.id: cell.colour for cell in organ.cells}
    colours = [colours_dict[label] for label in data.x]

    # Specify save graph vis location
    save_path = save_dir / "delaunay"
    save_path.mkdir(parents=True, exist_ok=True)

    delaunay = make_delaunay_graph(data)
    print(f"Plotting...")

    fig = plt.figure(figsize=(8, 8), dpi=150)
    plt.triplot(
        data.pos[:, 0], data.pos[:, 1], delaunay.simplices, linewidth=0.3, color="black"
    )
    plt.scatter(data.pos[:, 0], data.pos[:, 1], s=2, zorder=1000, c=colours)
    plt.gca().invert_yaxis()
    plt.axis("off")
    fig.tight_layout()

    plot_name = f"{plot_name}.png"
    plt.savefig(save_path / plot_name)
    print(f"Plot saved to {save_path / plot_name}")


def _visualize_points(
    organ, save_path, pos, labels=None, edge_index=None, edge_weight=None
):
    colours_dict = {cell.id: cell.colour for cell in organ.cells}
    colours = [colours_dict[label] for label in labels]

    fig = plt.figure(figsize=(8, 8), dpi=150)

    if edge_index is not None:
        line_collection = []
        for i, (src, dst) in enumerate(edge_index.t().tolist()):
            src = pos[src].tolist()
            dst = pos[dst].tolist()
            line_collection.append((src, dst))
        lc = LineCollection(
            line_collection,
            linewidths=0.5,
            colors=[str(weight) for weight in edge_weight.t()[0].tolist()],
        )
        ax = plt.gca()
        ax.add_collection(lc)
        ax.autoscale()

    plt.scatter(pos[:, 0], pos[:, 1], s=2, zorder=1000, c=colours)
    plt.gca().invert_yaxis()
    plt.axis("off")
    fig.tight_layout()
    plt.savefig(save_path)


def _vis_with_networkx(data, organ):
    print(f"Generating graph with networkx")
    print(f"Plotting...")

    labels = nx.get_node_attributes(data, "x").values()
    colours_dict = {cell.id: cell.colour for cell in organ.cells}
    colours = [colours_dict[label] for label in labels]
    fig = plt.figure(figsize=(8, 8), dpi=150)
    nx.draw(
        data, cmap=plt.get_cmap("Set1"), node_color=colours, node_size=75, linewidths=6
    )
    plt.show()


if __name__ == "__main__":
    typer.run(main)
