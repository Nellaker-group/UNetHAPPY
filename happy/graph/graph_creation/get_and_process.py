import os

import numpy as np
import pandas as pd

from hdf5 import get_embeddings_file, HDF5Dataset
from projects.placenta.graphs.processing.process_knots import process_knts


def get_and_process_raw_data(
    project_name, organ, project_dir, run_id, graph_params, tissue_label_tsv
):
    hdf5_data, tissue_class = get_raw_data(
        project_name, organ, project_dir, run_id, graph_params, tissue_label_tsv
    )
    hdf5_data, tissue_class = process_raw_data(
        organ,
        hdf5_data,
        tissue_class,
        graph_params["group_knts"],
        graph_params["random_remove"],
        graph_params["top_conf"],
    )
    return hdf5_data, tissue_class


def get_raw_data(
    project_name,
    organ,
    project_dir,
    run_id,
    graph_params,
    tissue_label_tsv,
):
    # Get training data from hdf5 files
    hdf5_data = get_hdf5_data(
        project_name,
        run_id,
        graph_params["x_min"],
        graph_params["y_min"],
        graph_params["width"],
        graph_params["height"],
    )
    # Get ground truth manually annotated data
    _, _, tissue_class = get_groundtruth_patch(
        organ,
        project_dir,
        graph_params["x_min"],
        graph_params["y_min"],
        graph_params["width"],
        graph_params["height"],
        tissue_label_tsv,
    )
    return hdf5_data, tissue_class


def process_raw_data(
    organ,
    hdf5_data,
    tissue_class=None,
    group_knts=True,
    random_remove=False,
    top_conf=False,
):
    # Covert isolated knts into syn and turn groups into a single knt point
    if group_knts:
        hdf5_data, tissue_class = process_knts(organ, hdf5_data, tissue_class)
    if top_conf:
        hdf5_data, tissues = confidence_filter(hdf5_data, 0.9, 1.0, tissue_class)
    # Remove a random percentage of the data
    if random_remove > 0.0:
        hdf5_data, tissue_class = random_filter(hdf5_data, random_remove, tissue_class)
    return hdf5_data, tissue_class


def random_filter(hdf5_data, percent_to_remove, tissues=None):
    hdf5_data, mask = hdf5_data.filter_randomly(percent_to_remove)
    # Remove points from tissue ground truth as well
    if tissues is not None:
        tissues = tissues[mask]
    print(f"Randomly removed {len(tissues) - sum(mask)} points")
    return hdf5_data, tissues


def confidence_filter(hdf5_data, min_conf, max_conf, tissues=None):
    hdf5_data, mask = hdf5_data.filter_by_confidence(min_conf, max_conf)
    # Remove points from tissue ground truth as well
    if tissues is not None:
        tissues = tissues[mask]
    return hdf5_data, tissues


def get_hdf5_data(
    project_name, run_id, x_min, y_min, width, height, tissue=False, verbose=True
):
    embeddings_path = get_embeddings_file(project_name, run_id, tissue)
    if verbose:
        print(f"Getting data from: {embeddings_path}")
        print(f"Using patch of size: x{x_min}, y{y_min}, w{width}, h{height}")
    # Get hdf5 datasets contained in specified box/patch of WSI
    hdf5_data = HDF5Dataset.from_path(embeddings_path)
    hdf5_data = hdf5_data.filter_by_patch(
        x_min, y_min, width, height
    ).sort_by_coordinates()
    return hdf5_data


def get_groundtruth_patch(organ, project_dir, x_min, y_min, width, height, annot_tsv):
    if not annot_tsv:
        print("No tissue annotation tsv supplied")
        return None, None, None
    tissue_label_path = project_dir / "results" / "tissue_annots" / annot_tsv
    if not os.path.exists(str(tissue_label_path)):
        print("No tissue label tsv found")
        return None, None, None

    ground_truth_df = pd.read_csv(tissue_label_path, sep="\t")
    xs = ground_truth_df["px"].to_numpy()
    ys = ground_truth_df["py"].to_numpy()
    tissue_classes = ground_truth_df["class"].to_numpy()

    if x_min == 0 and y_min == 0 and width == -1 and height == -1:
        sort_args = np.lexsort((ys, xs))
        tissue_ids = np.array(
            [organ.tissue_by_label(tissue_name).id for tissue_name in tissue_classes]
        )
        return xs[sort_args], ys[sort_args], tissue_ids[sort_args]

    mask = np.logical_and(
        (np.logical_and(xs > x_min, (ys > y_min))),
        (np.logical_and(xs < (x_min + width), (ys < (y_min + height)))),
    )
    patch_xs = xs[mask]
    patch_ys = ys[mask]
    patch_tissue_classes = tissue_classes[mask]

    patch_tissue_ids = np.array(
        [organ.tissue_by_label(tissue_name).id for tissue_name in patch_tissue_classes]
    )
    sort_args = np.lexsort((patch_ys, patch_xs))

    return patch_xs[sort_args], patch_ys[sort_args], patch_tissue_ids[sort_args]
