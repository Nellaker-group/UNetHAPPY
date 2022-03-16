import os

import typer
import pandas as pd

from happy.utils.utils import get_project_dir
from happy.organs.organs import get_organ


# TODO: this will have to be tweaked for the Towards extra folders
def main(
    project_name: str = typer.Option(...),
    organ_name: str = typer.Option(...),
    dataset_name: str = typer.Option(...),
    move_files: bool = False,
):
    project_dir = get_project_dir(project_name)
    cell_annot_dir = project_dir / "annotations" / "cell_class" / dataset_name
    cell_image_dir = project_dir / "datasets" / "cell_class" / dataset_name

    organ = get_organ(organ_name)
    all_cell_types = [cell.label for cell in organ.cells]

    split_types = ["train_cell.csv", "val_cell.csv", "test_cell.csv"]

    # Check that all values in the csvs are in the correct folders
    for split_type in split_types:
        print(f"{dataset_name} {split_type}:")
        df = pd.read_csv(cell_annot_dir / split_type, names=["path", "cell"])
        missing_images = []
        for index, image in df.iterrows():
            if not os.path.exists(project_dir / image["path"]):
                missing_images.append(image["path"])
        if len(missing_images) == 0:
            print("No missing images from csv file")
        else:
            print(missing_images)
            if move_files:
                pass
                # TODO: Find the missing images and move them to the correct folder

    # Check that there aren't missed images in the folders
    all_dfs = pd.concat(
        [
            pd.read_csv(cell_annot_dir / csv_file, names=["path", "cell"])
            for csv_file in split_types
        ]
    )
    all_images = {cell_type: [] for cell_type in all_cell_types}
    for cell_type in all_cell_types:
        if os.path.exists(cell_image_dir / cell_type):
            all_images[cell_type] = os.listdir(cell_image_dir / cell_type)
        else:
            all_images.pop(cell_type)
    missing_images = {cell_type: [] for cell_type in all_images}

    image_dir = f"datasets/cell_class/{dataset_name}"
    for cell_type in all_images:
        for image in all_images[cell_type]:
            image_path = f"{image_dir}/{cell_type}/{image}"
            if image_path not in all_dfs['path'].values:
                missing_images[cell_type].append(image)

    for cell_type in missing_images:
        number_missed = len(missing_images[cell_type])
        print(f"{cell_type} - {number_missed} images not in csvs")

    if move_files:
        for cell_type in missing_images:
            number_missed = len(missing_images[cell_type])
            if number_missed > 0:
                old_dir = cell_image_dir / cell_type
                move_to_dir = cell_image_dir / 'removed' / cell_type
                move_to_dir.mkdir(parents=True, exist_ok=True)
                for missed_image in missing_images[cell_type]:
                    old_path = old_dir / missed_image
                    new_path = move_to_dir / missed_image
                    os.rename(str(old_path.resolve()), str(new_path.resolve()))
                print(f"Moved {number_missed} {cell_type} images to {move_to_dir}")


if __name__ == "__main__":
    typer.run(main)
