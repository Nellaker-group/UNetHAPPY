from enum import Enum
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import typer
from matplotlib.collections import LineCollection
from torch_geometric.data import Data

import happy.db.eval_runs_interface as db
from happy.organs.organs import get_organ
from happy.hdf5.utils import (
    get_datasets_in_patch,
    filter_by_confidence,
    get_embeddings_file,
)
from happy.utils.utils import get_project_dir
from projects.placenta.graphs.graphs.create_graph import (
    make_k_graph,
    make_radius_k_graph,
    make_voronoi,
    make_delaunay_triangulation,
)


class MethodArg(str, Enum):
    k = "k"
    radius = "radius"
    voronoi = "voronoi"
    delaunay = "delaunay"
    all = "all"


def main(
    run_id: int = typer.Option(...),
    organ_name: str = "placenta",
    method: MethodArg = MethodArg.all,
    x_min: int = 0,
    y_min: int = 0,
    width: int = -1,
    height: int = -1,
    top_conf: bool = False,
    plot_edges: bool = False,
):
    """Generates a graph and saves it's visualisation. Node are coloured by cell type

    Args:
        run_id: id of the run which generated the embeddings file
        organ_name: name of the organ to get the cell colours
        method: graph creation method to use.
        x_min: min x coordinate for defining a subsection/patch of the WSI
        y_min: min y coordinate for defining a subsection/patch of the WSI
        width: width for defining a subsection/patch of the WSI. -1 for all
        height: height for defining a subsection/patch of the WSI. -1 for all
        top_conf: filter the nodes to only those >90% network confidence
        plot_edges: whether to plot edges or just points
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

    method = method.value
    if method == "k":
        vis_for_range_k(
            6, 7, data, plot_name, save_dir, organ, width, height, plot_edges
        )
    elif method == "radius":
        vis_for_range_radius(200, 260, 20, data, plot_name, save_dir, organ)
    elif method == "voronoi":
        vis_voronoi(data, plot_name, save_dir, organ)
    elif method == "delaunay":
        vis_delaunay(data, plot_name, save_dir, organ)
    elif method == "all":
        vis_for_range_k(
            6, 7, data, plot_name, save_dir, organ, width, height, plot_edges
        )
        vis_voronoi(data, plot_name, save_dir, organ)
        vis_delaunay(data, plot_name, save_dir, organ)
    else:
        raise ValueError(f"no such method: {method}")


def vis_for_range_k(
    k_start, k_end, data, plot_name, save_dir, organ, width, height, plot_edges=True
):
    if not plot_edges:
        edge_index = None
        edge_weight = None
    else:
        edge_index = data.edge_index
        edge_weight = data.edge_attr

    # Specify save graph vis location
    save_path = save_dir / "max_radius"
    save_path.mkdir(parents=True, exist_ok=True)

    # Generate vis for different values of k
    for k in range(k_start, k_end):
        data = make_k_graph(data, k)

        plot_name = f"k{k}_{plot_name}.png"
        print(f"Plotting...")
        visualize_points(
            organ,
            save_path / plot_name,
            data.pos,
            labels=data.x,
            edge_index=edge_index,
            edge_weight=edge_weight,
            width=width,
            height=height,
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
        visualize_points(
            organ,
            save_path / plot_name,
            data.pos,
            labels=data.x,
            edge_index=data.edge_index,
        )
        print(f"Plot saved to {save_path / plot_name}")


def vis_voronoi(data, plot_name, save_dir, organ, show_points=False):
    colours_dict = {cell.id: cell.colour for cell in organ.cells}
    colours = [colours_dict[label] for label in data.x]

    # Specify save graph vis location
    save_path = save_dir / "voronoi"
    save_path.mkdir(parents=True, exist_ok=True)

    vor = make_voronoi(data)
    print(f"Plotting...")

    point_size = 0.5 if len(vor.vertices) >= 10000 else 1

    fig = plt.figure(figsize=(8, 8), dpi=150)
    ax = plt.gca()
    finite_segments = []
    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0):
            vertices = vor.vertices[simplex]
            if (
                vertices[:, 0].min() >= vor.min_bound[0]
                and vertices[:, 0].max() <= vor.max_bound[0]
                and vertices[:, 1].min() >= vor.min_bound[1]
                and vertices[:, 1].max() <= vor.max_bound[1]
            ):
                finite_segments.append(vertices)
    if show_points:
        plt.scatter(
            vor.points[:, 0],
            vor.points[:, 1],
            marker=".",
            s=point_size,
            zorder=1000,
            c=colours,
        )
    else:
        xys = np.array(finite_segments).reshape(-1, 2)
        plt.scatter(xys[:, 0], xys[:, 1], marker=".", s=point_size, zorder=1000)
    ax.add_collection(
        LineCollection(
            finite_segments, colors="black", lw=0.5, alpha=0.6, linestyle="solid"
        )
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

    delaunay = make_delaunay_triangulation(data)
    print(f"Plotting...")

    point_size = 1 if len(delaunay.edges) >= 10000 else 2

    fig = plt.figure(figsize=(8, 8), dpi=150)
    plt.triplot(delaunay, linewidth=0.3, color="black")
    plt.scatter(
        data.pos[:, 0], data.pos[:, 1], marker=".", s=point_size, zorder=1000, c=colours
    )
    plt.gca().invert_yaxis()
    plt.axis("off")
    fig.tight_layout()

    plot_name = f"{plot_name}.png"
    plt.savefig(save_path / plot_name)
    print(f"Plot saved to {save_path / plot_name}")


def visualize_points(
    organ,
    save_path,
    pos,
    width=None,
    height=None,
    labels=None,
    edge_index=None,
    edge_weight=None,
    colours=None,
):
    if colours is None:
        colours_dict = {cell.id: cell.colour for cell in organ.cells}
        colours = [colours_dict[label] for label in labels]

    point_size = 1 if len(pos) >= 10000 else 2

    figsize = _calc_figsize(pos, width, height)

    fig = plt.figure(figsize=figsize, dpi=150)

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
    plt.scatter(
        pos[:, 0],
        pos[:, 1],
        marker=".",
        s=point_size,
        zorder=1000,
        c=colours,
        cmap="Spectral",
    )
    plt.gca().invert_yaxis()
    plt.axis("off")
    fig.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.0)


def _calc_figsize(pos, width, height):
    if width is None and height is None:
        return 8, 8
    if width == -1 and height == -1:
        pos_width = max(pos[:, 0]) - min(pos[:, 0])
        pos_height = max(pos[:, 1]) - min(pos[:, 1])
        ratio = pos_width / pos_height
        length = ratio * 8
        return length, 8
    else:
        ratio = width / height
        length = ratio * 8
        return length, 8


if __name__ == "__main__":
    typer.run(main)
