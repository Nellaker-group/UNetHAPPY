from typing import List
import os

import typer
import pandas as pd

from happy.utils.utils import get_project_dir


def main(
    project_name: str = typer.Option(...),
    dataset_names: List[str] = typer.Option([]),
):
    project_dir = get_project_dir(project_name)
    nuclei_annot_dir = project_dir / "annotations" / "nuclei"

    for dataset in dataset_names:
        print(f"Dataset: {dataset}")
        dataset_path = nuclei_annot_dir / dataset
        annot_files = [
            f for f in os.listdir(dataset_path) if os.path.isfile(dataset_path / f)
        ]
        for annot_file in annot_files:
            print(f"Split file: {annot_file}")
            df = pd.read_csv(
                dataset_path / annot_file,
                names=["path", "x1", "y1", "x2", "y2", "class"],
            )
            grouped_df = df.groupby(["path"]).size().reset_index(name="counts")

            print(grouped_df)


if __name__ == "__main__":
    typer.run(main)
