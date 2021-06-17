from pathlib import Path

import typer
import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch_cluster import knn_graph, radius_graph
from torch_geometric.data import Data
from torch_geometric.transforms import Distance

import nucnet.db.eval_runs_interface as db
from nucnet.hdf5.utils import get_datasets_in_patch, filter_by_confidence
from nucnet.utils.enum_args import OrganArg


def main(
    run_id: int = -1,
    x_min: int = 0,
    y_min: int = 0,
    width: int = -1,
    height: int = -1,
    top_conf: bool = False,
    organ: OrganArg = OrganArg.placenta,
):
    """Generates a graph and saves it's visualisation. Node are coloured by cell type

    Args:
        run_id: id of the run which generated the embeddings file
        x_min: min x coordinate for defining a subsection/patch of the WSI
        y_min: min y coordinate for defining a subsection/patch of the WSI
        width: width for defining a subsection/patch of the WSI. -1 for all
        height: height for defining a subsection/patch of the WSI. -1 for all
        top_conf: filter the nodes to only those >90% network confidence
    """
    db.init()
    eval_run = db.get_eval_run_by_id(run_id)

    # Get path to embeddings hdf5 files
    embeddings_root = Path(__file__).parent.parent / "results" / "embeddings"
    embeddings_path = eval_run.embeddings_path
    embeddings_dir = embeddings_root / embeddings_path
    print(f"Getting data from: {embeddings_path}")

    # Get hdf5 datasets contained in specified box/patch of WSI
    predictions, embeddings, coords, confidence = get_datasets_in_patch(
        embeddings_dir, x_min, y_min, width, height
    )
    print(f"Data loaded with {len(predictions)} nodes")

    if top_conf:
        predictions, embeddings, coords, confidence = filter_by_confidence(
            predictions, embeddings, coords, confidence, 0.9, 1.0
        )

    # Make graph data object
    data = Data(x=predictions, pos=torch.Tensor(coords.astype("int32")))

    conf_str = "_top_conf" if top_conf else ""
    plot_name = f"x{x_min}_y{y_min}{conf_str}"
    vis_for_range_k(5, 6, data, plot_name, embeddings_path, width, height, organ.value)


def vis_for_range_k(k_start, k_end, data, plot_name, file_path, width, height, organ):
    # Specify save graph vis location
    save_dir = file_path.split("run")[0]
    save_path = (
        Path(__file__).parent.parent
        / "visualisations"
        / "graphs"
        / save_dir
        / f"w{width}_h{height}"
        / "max_radius"
    )
    save_path.mkdir(parents=True, exist_ok=True)

    # Generate vis for different values of k
    for k in range(k_start, k_end):
        print(f"Generating graph for k={k}")
        data.edge_index = knn_graph(data.pos, k=k + 1, loop=True)
        get_edge_distance_weights = Distance(cat=False)
        data = get_edge_distance_weights(data)
        print(f"Graph made with {len(data.edge_index[0])} edges!")

        print(f"Plotting...")
        visualize_points(
            organ,
            save_path / plot_name,
            data.pos,
            labels=data.x,
            edge_index=data.edge_index,
            edge_weight=data.edge_attr,
        )
        plot_name = f"k{k}_{plot_name}.png"
        print(f"Plot saved to {save_path / plot_name}")


def visualize_points(
    organ, save_path, pos, labels=None, edge_index=None, edge_weight=None
):
    colours_dict = {cell.id: cell.colour for cell in organ.cells}
    colours = [colours_dict[label] for label in labels]
    fig = plt.figure(figsize=(8, 8), dpi=150)
    if edge_index is not None:
        for i, (src, dst) in enumerate(edge_index.t().tolist()):
            src = pos[src].tolist()
            dst = pos[dst].tolist()
            plt.plot(
                [src[0], dst[0]],
                [src[1], dst[1]],
                linewidth=0.5,
                color=str(edge_weight[i].tolist()[0]),
            )
    plt.scatter(pos[:, 0], pos[:, 1], s=1, zorder=1000, c=colours)
    plt.gca().invert_yaxis()
    plt.axis("off")
    fig.tight_layout()
    plt.savefig(save_path)


# Example: vis_for_range_radius(200, 260, 20, data, x_min, y_min, embeddings_path, w, h)
def vis_for_range_radius(
    rad_start, rad_end, k, data, x_min, y_min, file_path, width, height
):
    data.edge_index = knn_graph(data.pos, k=5)
    # Specify save graph vis location
    save_dir = file_path.split("run")[0]

    # Generate vis for different values of k
    for radius in range(rad_start, rad_end, 10):
        save_path = (
            Path(__file__).parent.parent
            / "visualisations"
            / "graphs"
            / save_dir
            / f"w{width}_h{height}"
            / f"radius_{radius}"
        )
        save_path.mkdir(parents=True, exist_ok=True)

        print(f"Generating graph for radius={radius} and k={k}")
        data.edge_index = radius_graph(data.pos, r=radius, max_num_neighbors=k)
        print("Graph made!")

        print(f"Plotting...")
        save_file = f"k{k}_x{x_min}_y{y_min}.png"
        visualize_points(
            save_path / save_file,
            data.pos,
            labels=data.x,
            edge_index=data.edge_index,
        )
        print(f"Plot saved to {save_path / save_file}")


def vis_networkx(data, organ):
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
