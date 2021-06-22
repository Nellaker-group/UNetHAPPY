from pathlib import Path

import numpy as np
import h5py

import happy.db.eval_runs_interface as db


def get_embeddings_file(project_name, run_id):
    db.init()
    embeddings_dir = (
        Path(__file__).parent.parent.parent
        / "projects"
        / project_name
        / "results"
        / "embeddings"
    )
    embeddings_path = db.get_embeddings_path(run_id, embeddings_dir)
    return embeddings_dir / embeddings_path


def get_hdf5_datasets(file_path, start, num_points):
    with h5py.File(file_path, "r") as f:
        subset_start = (
            int(len(f["predictions"]) * start) if 1 > start > 0 else int(start)
        )
        subset_end = (
            len(f["predictions"]) if num_points == -1 else subset_start + num_points
        )
        print(f"Getting {subset_end - subset_start} datapoints from hdf5")
        predictions = f["predictions"][subset_start:subset_end]
        embeddings = f["embeddings"][subset_start:subset_end]
        coords = f["coords"][subset_start:subset_end]
        confidence = f["confidence"][subset_start:subset_end]
        return predictions, embeddings, coords, confidence, subset_start, subset_end


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
        cell_class = metric_start
        label_map = {cell.label: cell.id for cell in organ.cells}
        filtered_embeddings = embeddings[predictions == label_map[cell_class]]
        filtered_predictions = predictions[predictions == label_map[cell_class]]
        filtered_confidence = confidence[predictions == label_map[cell_class]]
        filtered_coords = coords[predictions == label_map[cell_class]]
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


def get_datasets_in_patch(file_path, x_min, y_min, width, height):
    predictions, embeddings, coords, confidence, _, _ = get_hdf5_datasets(
        file_path, 0, -1
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
