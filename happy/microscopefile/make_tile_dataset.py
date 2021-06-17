import pandas as pd
from PIL import Image
from os import listdir
import random
import numpy as np

from happy.db.msfile_interface import get_msfile_by_run


# Generate 1600x1200 training images for nuclei detector
def make_nuclei_images(run_id, save_path, path_to_coords):
    coords = pd.read_csv(path_to_coords)
    xs = coords.bx.unique()
    ys = coords.by.unique()
    _save_pngs(run_id, save_path, (xs, ys))


# Generate 200x200 training images for cell classifier
def make_cell_images(run_id, save_path, path_to_coords):
    coords = pd.read_csv(path_to_coords)
    cell_classes = coords["class"]
    xs = coords["bx"] + coords["px"]
    ys = coords["by"] + coords["py"]
    _save_pngs(
        run_id,
        save_path,
        (xs, ys),
        width=200,
        height=200,
        cell_classes=cell_classes.tolist(),
    )


# Generate ground truth annotation csvs for nuclei detector.
def make_nuclei_annotations(run_id, save_path, coords_dir, path_to_images, split=None):
    file = get_msfile_by_run(run_id)

    # Get coordinates of all annotated files in directory
    coord_files = [f"{coords_dir}{f}" for f in listdir(coords_dir)]
    slide_numbers = [
        coord_file.split("/")[-1].split("_")[0] for coord_file in coord_files
    ]

    dfs = [pd.read_csv(f) for f in coord_files]
    for i, df in enumerate(dfs):
        df["slide_number"] = slide_numbers[i]

    coords = pd.concat(dfs, ignore_index=True)
    xs, ys = coords["px"], coords["py"]

    # Scale coordinates to model expected size (1600x1200)
    xs = (xs / file.rescale_ratio).astype(int)
    ys = (ys / file.rescale_ratio).astype(int)

    # Turns centre annotations into bounding boxes
    x1s = (xs - 40).astype(int)
    y1s = (ys - 40).astype(int)
    x2s = (xs + 40).astype(int)
    y2s = (ys + 40).astype(int)

    # Clip bounding boxes at image edges to size of image edge
    x1s.loc[x1s < 0] = 0
    y1s.loc[y1s < 0] = 0
    x2s.loc[x2s > 1600] = 1600
    y2s.loc[y2s > 1200] = 1200

    # Creates the image name for each annotation/image.
    # Needs to duplicate the image name for each nucleus
    coords["image_names"] = (
        coords.slide_number.map(str)
        + "_x"
        + coords.bx.map(str)
        + "_y"
        + coords.by.map(str)
        + ".png"
    )

    # Add the relative dataset path
    dataset_dir = path_to_images.split("/")[-2]
    image_paths = [
        f"Datasets/Nuclei/{dataset_dir}/{name}" for name in coords.image_names
    ]
    # Class is all nuclei here
    class_names = ["nucleus" for _ in range(len(xs))]

    # DataFrame with annotations in correct format
    annotation_df = pd.DataFrame(
        {
            "path": image_paths,
            "x1": x1s,
            "y1s": y1s,
            "x2s": x2s,
            "y2s": y2s,
            "class": class_names,
        }
    )

    # Either put all annotations into a file or make three separate train/val/test split files
    if not split:
        annotation_df.to_csv(f"{save_path}all_nuclei.csv", header=False, index=False)
    else:
        unique_tiles = annotation_df.path.unique().copy()
        num_val, num_test = _calc_splits(split, len(unique_tiles))
        # Randomly assign the tile names to each set
        random.shuffle(unique_tiles)
        _save_splits(
            "nuclei", unique_tiles, num_val, num_test, annotation_df, save_path
        )


# Generate ground truth annotation csvs for cell classifier
def make_cell_annotations(save_path, path_to_images, split=None):
    # Creates the image path for each annotation/image
    image_dirs = [
        f"{path_to_images}{f}" for f in listdir(path_to_images) if f[0] != "."
    ]
    dataset_dir = path_to_images.split("/")[-2]

    all_paths = []
    all_cells = []
    for cell_class_path in image_dirs:
        cell_class = cell_class_path.split("/")[-1]
        file_paths = [
            f"Datasets/CellClass/{dataset_dir}/{cell_class}/{f}"
            for f in listdir(cell_class_path)
        ]
        cell_class = [cell_class for _ in range(len(listdir(cell_class_path)))]
        all_paths.append(file_paths)
        all_cells.append(cell_class)
    all_paths = [i for sublist in all_paths for i in sublist]
    all_cells = [i for sublist in all_cells for i in sublist]

    # DataFrame with annotations in correct format
    annotation_df = pd.DataFrame({"path": all_paths, "class": all_cells})

    # Either put all annotations into a file or make three separate train/val/test split files
    if not split:
        annotation_df.to_csv(f"{save_path}all_cell.csv", header=False, index=False)
    else:
        num_val, num_test = _calc_splits(split, len(annotation_df))
        # Randomly assign the tile names to each set
        random_images = annotation_df.path.to_numpy().copy()
        random.shuffle(random_images)
        _save_splits("cell", random_images, num_val, num_test, annotation_df, save_path)


# Generate ground truth annotation csvs for empty tiles
def make_empty_annotations(path_to_images, save_path, split=None):
    image_names = [f"Datasets/Nuclei/empty/{f}" for f in listdir(path_to_images)]
    empty_annot = np.array(["" for _ in range(len(image_names))])

    annotation_df = pd.DataFrame(
        {
            "path": image_names,
            "x1": empty_annot,
            "y1s": empty_annot,
            "x2s": empty_annot,
            "y2s": empty_annot,
            "class": empty_annot,
            "extra": empty_annot,
        }
    )

    # Either put all annotations into a file or make three separate train/val/test split files
    if not split:
        annotation_df.to_csv(f"{save_path}all_cell.csv", header=False, index=False)
    else:
        num_val, num_test = _calc_splits(split, len(annotation_df))
        # Randomly assign the tile names to each set
        random_images = annotation_df.path.to_numpy().copy()
        random.shuffle(random_images)
        _save_splits(
            "nuclei", random_images, num_val, num_test, annotation_df, save_path
        )


# Evenly split the remainder into test and val.
# If split results in odd number then add the extra to test set
def _calc_splits(split, total_len):
    num_val_test = int(total_len * split)
    num_val = int(num_val_test / 2)
    num_test = (
        int(num_val_test / 2) if (num_val_test % 2) == 0 else int(num_val_test / 2) + 1
    )
    return num_val, num_test


# Splits the annotations into different datasets
def _save_splits(object_type, images, num_val, num_test, df, save_path):
    val = images[0:num_val]
    test = images[num_val : num_val + num_test]
    train = images[num_val + num_test :]
    # Make separate dataframes from the original
    val_df = df.loc[df["path"].isin(val)]
    test_df = df.loc[df["path"].isin(test)]
    train_df = df.loc[df["path"].isin(train)]
    # Save csvs
    val_df.to_csv(f"{save_path}val_{object_type}.csv", header=False, index=False)
    test_df.to_csv(f"{save_path}test_{object_type}.csv", header=False, index=False)
    train_df.to_csv(f"{save_path}train_{object_type}.csv", header=False, index=False)


# Saves a set of images based on coordinate and image size at save path location
def _save_pngs(run_id, save_path, coords, width=None, height=None, cell_classes=None):
    file = get_msfile_by_run(run_id)
    slide_name = file.slide_path.split("/")[-1]
    slide_number = slide_name.split("-")[0]

    width = width if width else file.target_tile_width
    height = height if height else file.target_tile_height
    xs, ys = coords
    xs = (xs - (100 * file.rescale_ratio)).astype(int) if cell_classes else xs
    ys = (ys - (100 * file.rescale_ratio)).astype(int) if cell_classes else ys

    for i in range(len(xs)):
        print(f"number: {i}")
        x, y = xs[i], ys[i]
        final_save_path = (
            f"{save_path}{cell_classes[i]}/" if cell_classes else save_path
        )

        output_array = file.get_tile_by_coords(x, y, width, height)
        im = Image.fromarray(output_array.astype("uint8"))
        im.save(f"{final_save_path}/{slide_number}_x{x}_y{y}.png")
        print(f"generated: {slide_number}_x{x}_y{y}.png")
