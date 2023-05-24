from pathlib import Path

import numpy as np
import h5py

import happy.db.eval_runs_interface as db


def get_embeddings_file(project_name, run_id, tissue=False):
    embeddings_dir = (
        Path(__file__).parent.parent.parent
        / "projects"
        / project_name
        / "results"
        / "embeddings"
    )
    embeddings_path = db.get_embeddings_path(run_id, embeddings_dir)
    if tissue:
        file_name = f"{embeddings_path.name.split('.hdf5')[0]}_tissues.hdf5"
        embeddings_path = embeddings_path.parent / file_name
    return embeddings_dir / embeddings_path


def get_hdf5_datasets(file_path, start, num_points, verbose=True):
    with h5py.File(file_path, "r") as f:
        subset_start = (
            int(len(f["predictions"]) * start) if 1 > start > 0 else int(start)
        )
        subset_end = (
            len(f["predictions"]) if num_points == -1 else subset_start + num_points
        )
        if verbose:
            print(f"Getting {subset_end - subset_start} datapoints from hdf5")
        predictions = f["predictions"][subset_start:subset_end]
        embeddings = f["embeddings"][subset_start:subset_end]
        coords = f["coords"][subset_start:subset_end]
        confidence = f["confidence"][subset_start:subset_end]
        return predictions, embeddings, coords, confidence, subset_start, subset_end


def get_tissue_hdf5_datasets(file_path, start, num_points, verbose=True):
    with h5py.File(file_path, "r") as f:
        subset_start = (
            int(len(f["predictions"]) * start) if 1 > start > 0 else int(start)
        )
        subset_end = (
            len(f["predictions"]) if num_points == -1 else subset_start + num_points
        )
        if verbose:
            print(f"Getting {subset_end - subset_start} datapoints from hdf5")
        cell_predictions = f["cell_predictions"][subset_start:subset_end]
        cell_embeddings = f["cell_embeddings"][subset_start:subset_end]
        coords = f["coords"][subset_start:subset_end]
        cell_confidence = f["cell_confidence"][subset_start:subset_end]
        tissue_predictions = f["tissue_predictions"][subset_start:subset_end]
        tissue_embeddings = f["tissue_embeddings"][subset_start:subset_end]
        tissue_confidence = f["tissue_confidence"][subset_start:subset_end]
        return (
            cell_predictions,
            cell_embeddings,
            coords,
            cell_confidence,
            tissue_predictions,
            tissue_embeddings,
            tissue_confidence,
            subset_start,
            subset_end,
        )


def filter_hdf5(
    organ, file_path, start, num_points, metric_type, metric_start, metric_end=None
):
    (
        predictions,
        embeddings,
        coords,
        confidence,
        subset_start,
        subset_end,
    ) = get_hdf5_datasets(file_path, start, num_points)

    if metric_type == "cell_class":
        (
            filtered_predictions,
            filtered_embeddings,
            filtered_coords,
            filtered_confidence,
        ) = filter_by_cell_type(
            predictions, embeddings, coords, confidence, metric_start, organ
        )
    elif metric_type == "confidence":
        min_conf = metric_start
        max_conf = metric_end
        (
            filtered_predictions,
            filtered_embeddings,
            filtered_coords,
            filtered_confidence,
        ) = filter_by_confidence(
            predictions, embeddings, coords, confidence, min_conf, max_conf
        )
    else:
        raise ValueError(f"[{metric_type}] is not a valid metric type")

    num_filtered = len(filtered_embeddings)
    print(f"num of cells: {num_filtered}")
    return (
        filtered_predictions,
        filtered_embeddings,
        filtered_coords,
        filtered_confidence,
        subset_start,
        subset_end,
        num_filtered,
    )


def get_datasets_in_patch(file_path, x_min, y_min, width, height, verbose=True):
    predictions, embeddings, coords, confidence, _, _ = get_hdf5_datasets(
        file_path, 0, -1, verbose=verbose
    )

    if x_min == 0 and y_min == 0 and width == -1 and height == -1:
        return predictions, embeddings, coords, confidence

    mask = np.logical_and(
        (np.logical_and(coords[:, 0] > x_min, (coords[:, 1] > y_min))),
        (
            np.logical_and(
                coords[:, 0] < (x_min + width), (coords[:, 1] < (y_min + height))
            )
        ),
    )

    patch_coords = coords[mask]
    patch_predictions = predictions[mask]
    patch_embeddings = embeddings[mask]
    patch_confidence = confidence[mask]

    return patch_predictions, patch_embeddings, patch_coords, patch_confidence


def filter_by_confidence(
    predictions, embeddings, coords, confidence, min_conf, max_conf
):
    filtered_embeddings = embeddings[
        np.logical_and((confidence >= min_conf), (confidence <= max_conf))
    ]
    filtered_predictions = predictions[
        np.logical_and((confidence >= min_conf), (confidence <= max_conf))
    ]
    filtered_confidence = confidence[
        np.logical_and((confidence >= min_conf), (confidence <= max_conf))
    ]
    filtered_coords = coords[
        np.logical_and((confidence >= min_conf), (confidence <= max_conf))
    ]
    return (
        filtered_predictions,
        filtered_embeddings,
        filtered_coords,
        filtered_confidence,
    )


def filter_by_cell_type(predictions, embeddings, coords, confidence, cell_type, organ):
    label_map = {cell.label: cell.id for cell in organ.cells}
    filtered_embeddings = embeddings[predictions == label_map[cell_type]]
    filtered_predictions = predictions[predictions == label_map[cell_type]]
    filtered_confidence = confidence[predictions == label_map[cell_type]]
    filtered_coords = coords[predictions == label_map[cell_type]]

    return (
        filtered_predictions,
        filtered_embeddings,
        filtered_coords,
        filtered_confidence,
    )


def filter_randomly(predictions, embeddings, coords, confidence, percent_to_remove):
    num_to_remove = int(len(predictions) * percent_to_remove)
    indices = np.random.choice(
        np.arange(len(predictions)), num_to_remove, replace=False
    )
    filtered_predictions = np.delete(predictions, indices)
    filtered_embeddings = np.delete(embeddings, indices, axis=0)
    filtered_confidence = np.delete(confidence, indices)
    filtered_coords = np.delete(coords, indices, axis=0)
    return (
        filtered_predictions,
        filtered_embeddings,
        filtered_coords,
        filtered_confidence,
        indices,
    )
