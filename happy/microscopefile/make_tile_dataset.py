import random
from os import listdir
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from PIL import Image
import typer

from happy.microscopefile.reader import Reader
import happy.db.eval_runs_interface as db


def main(
    project_name: str = typer.Option(...),
    dataset_type: str = typer.Option(...),
    dataset_name: str = typer.Option(...),
    coords_dir: Path = typer.Option(
        ..., exists=True, file_okay=False, dir_okay=True, resolve_path=True
    ),
    slide_ids: List[int] = typer.Option([]),
    coord_file_names: List[str] = typer.Option(...),
    target_pixel_size: float = 0.1109,
    tile_width: int = 1600,
    tile_height: int = 1200,
    split: Optional[float] = None,
):
    """Creates directories of images and annotation files for one dataset.

    Requires csv files generated by QuPath Groovy scripts. Processes multiple slides at
    once but they must be part of the same dataset. Order of inputted slide ids must
    match the order of inputted coordinate file names.

    NOTE: This will create a new annotation csv each time it is run and so will
    overwrite any existing csvs with the same name. Either move/rename those files
    or setup a different dataset directory for each new set of annotations. All
    annotated slides are expected to be part of the same dataset and so passed in
    together.

    Args:
        project_name: name of the project dir to save results to
        dataset_type: one of 'nuclei' or 'cell_class'
        dataset_name: a name for referencing this exact dataset to be generated
        coords_dir: path to directory containing Groovy generated csv files
        slide_ids: ids of slides for which there are csv files
        coord_file_names: names of Groovy csv files
        target_pixel_size: pixel size the original model was trained on for rescaling
        tile_width: width of tiles the model expects
        tile_height: height of tiles the model expects
        split: proportion of training data split, rest goes equally into val and test
    """
    db.init()

    save_path = (
        Path(__file__).absolute().parent.parent.parent
        / "projects"
        / project_name
        / "datasets"
        / dataset_type
        / dataset_name
    )

    slides = [db.get_slide_by_id(slide_id) for slide_id in slide_ids]
    paths_to_coords = [coords_dir / file_name for file_name in coord_file_names]

    if dataset_type == "nuclei":
        make_nuclei_images(
            save_path,
            slides,
            target_pixel_size,
            tile_width,
            tile_height,
            paths_to_coords,
        )
        make_nuclei_annotations(
            save_path, slides, target_pixel_size, paths_to_coords, split
        )
    elif dataset_type == "cell_class":
        make_cell_images(save_path, slides, target_pixel_size, paths_to_coords)
        make_cell_annotations(save_path, split)
    elif dataset_type == "empty":
        make_nuclei_images(
            save_path,
            slides,
            target_pixel_size,
            tile_width,
            tile_height,
            paths_to_coords,
        )
        make_empty_annotations(save_path, split)
    else:
        raise ValueError(f"No such dataset type supported: {dataset_type}")


# Generate training images for nuclei detector
def make_nuclei_images(
    save_path, slides, target_pixel_size, target_width, target_height, paths_to_coords
):
    for i, slide in enumerate(slides):
        coords = pd.read_csv(paths_to_coords[i])
        xs = coords.bx.unique()
        ys = coords.by.unique()
        _save_pngs(
            save_path, slide, (xs, ys), target_pixel_size, target_width, target_height
        )


# Generate 200x200 training images for cell classifier
def make_cell_images(save_path, slides, target_pixel_size, paths_to_coords):
    for i, slide in enumerate(slides):
        coords = pd.read_csv(paths_to_coords[i])
        cell_classes = coords["class"]
        xs = coords["bx"] + coords["px"]
        ys = coords["by"] + coords["py"]
        _save_pngs(
            save_path,
            slide,
            (xs, ys),
            target_pixel_size,
            target_width=200,
            target_height=200,
            cell_classes=cell_classes.tolist(),
        )


# Generate ground truth annotation csvs for nuclei detector.
def make_nuclei_annotations(
    save_path,
    slides,
    target_pixel_size,
    paths_to_coords,
    split=None,
):
    rescale_ratio = target_pixel_size / slides[0].pixel_size

    # Get coordinates of all annotated files in directory
    slide_numbers = [
        str(coord_file).split("/")[-1].split("_")[0] for coord_file in paths_to_coords
    ]
    dfs = [pd.read_csv(f) for f in paths_to_coords]
    for i, df in enumerate(dfs):
        df["slide_number"] = slide_numbers[i]

    coords = pd.concat(dfs, ignore_index=True)
    xs, ys = coords["px"], coords["py"]

    # Scale coordinates to model expected size (1600x1200)
    xs = (xs / rescale_ratio).astype(int)
    ys = (ys / rescale_ratio).astype(int)

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
    image_paths = [Path(*save_path.parts[-3:]) / name for name in coords.image_names]
    # Class is just nuclei here
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

    # Edit the save path so that it saves to /annotations rather than /datasets
    annotation_save_path = _get_annotation_path(save_path)

    # Either all annotations in one file or three separate train/val/test split files
    if not split:
        annotation_df.to_csv(
            annotation_save_path / "all_nuclei.csv", header=False, index=False
        )
    else:
        unique_tiles = annotation_df.path.unique().copy()
        num_val, num_test = _calc_splits(split, len(unique_tiles))
        # Randomly assign the tile names to each set
        random.shuffle(unique_tiles)
        _save_splits(
            "nuclei",
            unique_tiles,
            num_val,
            num_test,
            annotation_df,
            annotation_save_path,
        )


# Generate ground truth annotation csvs for cell classifier
def make_cell_annotations(save_path, split=None):
    # Creates the image path for each annotation/image
    image_dirs = [save_path / f for f in listdir(save_path) if f[0] != "."]

    all_paths = []
    all_cells = []
    for cell_class_dir in image_dirs:
        cell_class = cell_class_dir.parts[-1]
        file_paths = [
            Path(*cell_class_dir.parts[-4:]) / f for f in listdir(cell_class_dir)
        ]
        cell_class = [cell_class for _ in range(len(listdir(cell_class_dir)))]
        all_paths.extend(file_paths)
        all_cells.extend(cell_class)

    # DataFrame with annotations in correct format
    annotation_df = pd.DataFrame({"path": all_paths, "class": all_cells})

    # Edit the save path so that it saves to /annotations rather than /datasets
    annotation_save_path = _get_annotation_path(save_path)

    # Either all annotations in one file or three separate train/val/test split files
    if not split:
        annotation_df.to_csv(
            annotation_save_path / "all_cell.csv", header=False, index=False
        )
    else:
        num_val, num_test = _calc_splits(split, len(annotation_df))
        # Randomly assign the tile names to each set
        random_images = annotation_df.path.to_numpy().copy()
        random.shuffle(random_images)
        _save_splits(
            "cell",
            random_images,
            num_val,
            num_test,
            annotation_df,
            annotation_save_path,
        )


# Generate ground truth annotation csvs for empty tiles
def make_empty_annotations(save_path, split=None):
    image_names = [Path(*save_path.parts[-3:]) / f for f in listdir(save_path)]
    empty_annot = np.array(["" for _ in range(len(image_names))])

    annotation_df = pd.DataFrame(
        {
            "path": image_names,
            "x1": empty_annot,
            "y1s": empty_annot,
            "x2s": empty_annot,
            "y2s": empty_annot,
            "class": empty_annot,
        }
    )

    # Edit the save path so that it saves to /annotations rather than /datasets
    annotation_save_path = _get_annotation_path(save_path)

    # Either all annotations in one file or three separate train/val/test split files
    if not split:
        print(annotation_save_path)
        annotation_df.to_csv(
            annotation_save_path / "all_nuclei.csv", header=False, index=False
        )
    else:
        num_val, num_test = _calc_splits(split, len(annotation_df))
        # Randomly assign the tile names to each set
        random_images = annotation_df.path.to_numpy().copy()
        random.shuffle(random_images)
        _save_splits(
            "nuclei",
            random_images,
            num_val,
            num_test,
            annotation_df,
            annotation_save_path,
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
    val_df.to_csv(save_path / f"val_{object_type}.csv", header=False, index=False)
    test_df.to_csv(save_path / f"test_{object_type}.csv", header=False, index=False)
    train_df.to_csv(save_path / f"train_{object_type}.csv", header=False, index=False)


# TODO: Remove the string manipulation for getting the slide_number (find a better identifier)
# Saves a set of images based on coordinate and image size at save path location
def _save_pngs(
    save_path,
    slide,
    coords,
    target_pixel_size,
    target_width,
    target_height,
    cell_classes=None,
):
    slide_name = slide.slide_name
    slide_number = slide_name.split("-")[0]
    rescale_ratio = target_pixel_size / slide.pixel_size

    slide_path = str(Path(slide.lab.slides_dir) / slide_name)
    reader = Reader.new(slide_path, slide.lvl_x)

    xs, ys = coords
    xs = (xs - (100 * rescale_ratio)).astype(int) if cell_classes else xs
    ys = (ys - (100 * rescale_ratio)).astype(int) if cell_classes else ys

    for i in range(len(xs)):
        x, y = xs[i], ys[i]
        final_save_path = save_path / cell_classes[i] if cell_classes else save_path
        final_save_path.mkdir(parents=True, exist_ok=True)

        width = target_width * rescale_ratio
        height = target_height * rescale_ratio
        output_array = _get_image_crop(
            reader, x, y, width, height, target_width, target_height
        )
        im = Image.fromarray(output_array.astype("uint8"))
        im.save(final_save_path / f"{slide_number}_x{x}_y{y}.png")
        print(f"generated: {slide_number}_x{x}_y{y}.png")


def _get_image_crop(reader, x, y, width, height, target_width, target_height):
    img = reader.get_img(x, y, int(width), int(height))
    img = Image.fromarray(img.astype("uint8"))
    rescaled_img = img.resize([target_width, target_height])
    return np.asarray(rescaled_img)


def _get_annotation_path(dataset_path):
    save_path_parts = list(dataset_path.parts)
    save_path_parts[save_path_parts.index("datasets")] = "annotations"
    annotation_path = Path(*save_path_parts)
    annotation_path.mkdir(parents=True, exist_ok=True)
    return annotation_path


if __name__ == "__main__":
    typer.run(main)
