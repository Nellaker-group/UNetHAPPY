import os

import typer
import pandas as pd

from happy.utils.utils import get_project_dir
from happy.organs import get_organ


def main(
    project_name: str = typer.Option(...),
    organ_name: str = typer.Option(...),
    dataset_name: str = typer.Option(...),
    move_files: bool = False,
):
    """ Follows the paths in the csv files and checks that a corresponding image
    exists. If the image does not exist and move_files is True, then it will search
    all other cell type folders for the matching image and move it to the specified
    location.
    """
    project_dir = get_project_dir(project_name)
    cell_annot_dir = project_dir / "annotations" / "cell_class" / dataset_name
    cell_image_dir = project_dir / "datasets" / "cell_class" / dataset_name

    organ = get_organ(organ_name)
    all_cell_types = [cell.label for cell in organ.cells]

    split_types = ["train_cell.csv", "val_cell.csv", "test_cell.csv"]
    all_dfs = pd.concat(
        [
            pd.read_csv(cell_annot_dir / csv_file, names=["path", "cell"])
            for csv_file in split_types
        ]
    )

    # Check that all values in the csvs are in the correct folders
    print(f"{dataset_name}:")
    missing_images = []
    for index, image in all_dfs.iterrows():
        if not os.path.exists(project_dir / image["path"]):
            missing_images.append(image["path"])
    if len(missing_images) == 0:
        print("No images in csv files missing from folders")
    else:
        print(f"{len(missing_images)} images in csvs but missing from correct folder")
        if move_files:
            print("Moving missing images...")
            images_moved = 0
            for image in missing_images:
                image_name = image.split("/")[-1]
                new_cell_type = image.split("/")[-2]
                new_dir = project_dir / image.split(image_name)[0]
                new_dir.mkdir(parents=True, exist_ok=True)
                new_path = new_dir / image_name
                for cell_type in all_cell_types:
                    if cell_type == new_cell_type:
                        continue
                    if not os.path.exists(cell_image_dir / cell_type):
                        continue
                    old_path_split = image.split(new_cell_type)
                    old_path_split.insert(1, cell_type)
                    old_path = "".join(old_path_split)
                    full_old_path = (project_dir / old_path).resolve()
                    if os.path.exists(str(full_old_path)):
                        images_moved += 1
                        os.rename(str(full_old_path), str(new_path.resolve()))
            print(f"{images_moved} images moved to match path in csv files")


if __name__ == "__main__":
    typer.run(main)
