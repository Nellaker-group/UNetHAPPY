import typer
import sklearn.neighbors as sk
import matplotlib.pyplot as plt
import numpy as np

from happy.organs.organs import get_organ
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

    # Get path to embeddings hdf5 files
    embeddings_path = get_embeddings_file(project_name, run_id)
    # Get hdf5 datasets contained in specified box/patch of WSI
    all_predictions, all_embeddings, all_coords, all_confidence = get_datasets_in_patch(
        embeddings_path, x_min, y_min, width, height
    )

    _, _, _, _ = process_knt_cells(
        all_predictions,
        all_embeddings,
        all_coords,
        all_confidence,
        organ,
        radius,
        cut_off_count,
        width,
        height,
        plot=True,
    )


def process_knt_cells(
    all_predictions,
    all_embeddings,
    all_coords,
    all_confidence,
    organ,
    radius,
    cut_off_count,
    width,
    height,
    plot=False,
):
    # Filter by KNT cell type
    predictions, embeddings, coords, confidence = filter_by_cell_type(
        all_predictions, all_embeddings, all_coords, all_confidence, "KNT", organ
    )
    print(f"Data loaded with {len(predictions)} KNT cells")

    # Get indices of KNT cells in radius
    all_knt_inds = np.nonzero(all_predictions == 10)[0]
    unique_indices = _get_indices_in_radius(coords, radius)
    if plot:
        _plot_distances_to_nearest_neighbor([len(x) for x in unique_indices])

    # find KNT cells with no neighbors and turn them into SYN
    all_predictions, predictions = _convert_isolated_knt_into_syn(
        all_predictions, predictions, cut_off_count, unique_indices, all_knt_inds
    )

    # remove points with more than cut off neighbors and keep just one point
    (
        predictions,
        embeddings,
        coords,
        confidence,
        unique_inds_to_remove,
    ) = _cluster_knts_into_point(
        predictions, embeddings, coords, confidence, cut_off_count, unique_indices
    )

    if plot:
        print(f"Plotting...")
        visualize_points(
            organ,
            "plots/grouped_KNT.png",
            coords,
            labels=predictions,
            width=width,
            height=height,
        )

    inds_to_remove_from_total = all_knt_inds[unique_inds_to_remove]
    all_predictions = np.delete(all_predictions, inds_to_remove_from_total, axis=0)
    all_embeddings = np.delete(all_embeddings, inds_to_remove_from_total, axis=0)
    all_coords = np.delete(all_coords, inds_to_remove_from_total, axis=0)
    all_confidence = np.delete(all_confidence, inds_to_remove_from_total, axis=0)

    return all_predictions, all_embeddings, all_coords, all_confidence


def _plot_distances_to_nearest_neighbor(num_in_radius):
    plt.hist(num_in_radius, bins=100)
    plt.savefig("plots/num_in_radius_histogram.png")
    plt.clf()


def _get_indices_in_radius(coords, radius):
    tree = sk.KDTree(coords, metric="euclidean")
    all_nn_indices = tree.query_radius(coords, r=radius)
    # find indices of duplicate entries and remove duplicates
    unique_indices = np.unique(
        np.array([tuple(row) for row in all_nn_indices], dtype=object)
    )
    return unique_indices


def _convert_isolated_knt_into_syn(
    all_predictions, knt_predictions, cut_off_count, unique_indices, all_knt_inds
):
    lone_knt_indices = []
    lone_knt_indices_nested = [
        list(x) for x in unique_indices if len(x) <= cut_off_count
    ]
    for nested in lone_knt_indices_nested:
        lone_knt_indices.extend(nested)
    all_predictions[all_knt_inds[lone_knt_indices]] = 3
    knt_predictions[lone_knt_indices] = 3
    print(
        f"Converted {len(lone_knt_indices)} KNT cells with fewer than "
        f"{cut_off_count+1} neighbors into SYN"
    )
    return all_predictions, knt_predictions


def _cluster_knts_into_point(
    predictions, embeddings, coords, confidence, cut_off_count, unique_indices
):
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
    embeddings = np.delete(embeddings, unique_inds_to_remove, axis=0)
    coords = np.delete(coords, unique_inds_to_remove, axis=0)
    confidence = np.delete(confidence, unique_inds_to_remove, axis=0)
    return predictions, embeddings, coords, confidence, unique_inds_to_remove


if __name__ == "__main__":
    typer.run(main)
