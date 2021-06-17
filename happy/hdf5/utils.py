import numpy as np
import h5py


def filter_hdf5(
    file_path, start, num_points, metric_type, metric_start, metric_end=None
):
    predictions, embeddings, coords, confidence, subset_end = get_hdf5_datasets(
        file_path, start, num_points
    )

    if metric_type == "cell_class":
        cell_class = metric_start
        label_map = {"CYT": 0, "FIB": 1, "HOF": 2, "SYN": 3, "VEN": 4}
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
        raise ValueError(f"[{metric_type}] not a valid metric type")

    num_filtered = len(filtered_embeddings)
    print(f"num of cells: {num_filtered}")
    return (
        filtered_predictions,
        filtered_embeddings,
        filtered_coords,
        filtered_confidence,
        subset_end,
        num_filtered,
    )


def get_hdf5_datasets(file_path, start, num_points):
    with h5py.File(file_path, "r") as f:
        subset_start = start
        if num_points == -1:
            subset_end = len(f["predictions"])
        else:
            subset_end = subset_start + num_points

        predictions = f["predictions"][subset_start:subset_end]
        embeddings = f["embeddings"][subset_start:subset_end]
        coords = f["coords"][subset_start:subset_end]
        confidence = f["confidence"][subset_start:subset_end]
        return predictions, embeddings, coords, confidence, subset_end


def get_datasets_in_patch(file_path, x_min, y_min, width, height):
    predictions, embeddings, coords, confidence, _ = get_hdf5_datasets(file_path, 0, -1)

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
