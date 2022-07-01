import typer
import sklearn.neighbors as sk
import matplotlib.pyplot as plt
import numpy as np

from happy.organs.organs import get_organ
from happy.utils.utils import get_project_dir
import happy.db.eval_runs_interface as db
from happy.hdf5.utils import (
    get_datasets_in_patch,
    filter_by_cell_type,
    get_embeddings_file,
)
from projects.placenta.graphs.analysis.vis_graph_patch import visualize_points


def main(
    run_id: int = typer.Option(...),
    project_name: str = "placenta",
    x_min: int = 0,
    y_min: int = 0,
    width: int = -1,
    height: int = -1,
    radius: int = 100,
    cut_off_count: int = 3,
):
    # Create database connection
    db.init()

    organ = get_organ("placenta")
    project_dir = get_project_dir(project_name)

    # Get path to embeddings hdf5 files
    embeddings_path = get_embeddings_file(project_name, run_id)
    # Get hdf5 datasets contained in specified box/patch of WSI
    all_predictions, all_embeddings, all_coords, all_confidence = get_datasets_in_patch(
        embeddings_path, x_min, y_min, width, height
    )
    # Filter by KNT cell type
    all_knt_inds = np.nonzero(all_predictions == 10)[0]
    predictions, embeddings, coords, confidence = filter_by_cell_type(
        all_predictions, all_embeddings, all_coords, all_confidence, "KNT", organ
    )
    print(f"Data loaded with {len(predictions)} KNT cells")

    tree = sk.KDTree(coords, metric="euclidean")
    all_nn_indices = tree.query_radius(coords, r=radius)

    # find indices of duplicate entries and remove duplicates
    unique_indices = np.unique(
        np.array([tuple(row) for row in all_nn_indices], dtype=object)
    )
    unique_num_in_radius = [len(x) for x in unique_indices]
    plot_distances_to_nearest_neighbor(unique_num_in_radius)

    # remove KNT cells with no neighbors and turn them into SYN
    lone_knt_indices = []
    lone_knt_indices_nested = [list(x) for x in unique_indices if len(x) <= cut_off_count]
    for nested in lone_knt_indices_nested:
        lone_knt_indices.extend(nested)
    all_predictions[all_knt_inds[lone_knt_indices]] = 3
    predictions[lone_knt_indices] = 3
    print(f"Converted {len(lone_knt_indices)} KNT cells with few neighbors into SYN")

    # remove points with more than 4 neighbors and keep just one point
    remaining_inds = [x for x in unique_indices if len(x) > cut_off_count]
    # take all elements except the first to be removed from grouped KNT cells
    inds_to_remove = []
    for ind in remaining_inds:
        inds_to_remove.extend(list(ind[1:]))
    # remove duplicates inds
    inds_to_remove = np.array(inds_to_remove)
    unique_inds_to_remove = np.unique(inds_to_remove)

    # remove clustered
    predictions = np.delete(predictions, unique_inds_to_remove, axis=0)
    coords = np.delete(coords, unique_inds_to_remove, axis=0)

    print(f"Plotting...")
    visualize_points(
        organ,
        project_dir / "analysis" / "plots" / "grouped_KNT.png",
        coords,
        labels=predictions,
        width=width,
        height=height,
    )


def plot_distances_to_nearest_neighbor(num_in_radius):
    plt.hist(num_in_radius, bins=100)
    plt.savefig("plots/num_in_radius_histogram.png")
    plt.clf()


if __name__ == "__main__":
    typer.run(main)
