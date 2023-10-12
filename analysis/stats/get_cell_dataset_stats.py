from typing import List
import os

import typer
import pandas as pd
import seaborn as sns

from happy.utils.utils import get_project_dir
from happy.organs import get_organ


def main(
    project_name: str = typer.Option(...),
    organ_name: str = typer.Option(...),
    dataset_names: List[str] = typer.Option([]),
    plot_histogram: bool = True,
):
    project_dir = get_project_dir(project_name)
    cell_annot_dir = project_dir / "annotations" / "cell_class"

    organ = get_organ(organ_name)
    all_cell_types = [cell.label for cell in organ.cells]

    grouped_dfs = {}
    for dataset in dataset_names:
        print(f"Dataset: {dataset}")
        dataset_path = cell_annot_dir / dataset
        annot_files = [
            f for f in os.listdir(dataset_path) if os.path.isfile(dataset_path / f)
        ]
        if ".DS_Store" in annot_files:
            annot_files.remove(".DS_Store")
        for annot_file in annot_files:
            print(f"Split file: {annot_file}")
            df = pd.read_csv(dataset_path / annot_file, names=["path", "cell_class"])
            grouped_df = df.groupby(["cell_class"]).size().reset_index(name="counts")
            missing_cells = list(set(all_cell_types) - set(grouped_df["cell_class"]))
            missing_cells.sort()
            for cell in missing_cells:
                grouped_df.loc[len(grouped_df)] = pd.Series(
                    {"cell_class": cell, "counts": 0}
                )
            grouped_df.sort_values("cell_class", inplace=True, ignore_index=True)

            print(grouped_df)

            try:
                grouped_dfs[annot_file]["counts"] = (
                    grouped_dfs[annot_file]["counts"] + grouped_df["counts"]
                )
            except KeyError:
                grouped_dfs[annot_file] = grouped_df

    print("Combined Datasets")
    for annot_file in grouped_dfs:
        grouped_dfs[annot_file].sort_values(
            "cell_class", inplace=True, ignore_index=True
        )
        print(f"Split file: {annot_file}")
        print(grouped_dfs[annot_file])

        if plot_histogram:
            cell_colours = [cell.colour for cell in organ.cells]
            custom_palette = sns.set_palette(
                sns.color_palette(_colour_bars(cell_colours))
            )

            plot = sns.barplot(
                x=grouped_dfs[annot_file]["cell_class"],
                y=grouped_dfs[annot_file]["counts"],
                palette=custom_palette,
            )
            plot.figure.savefig(
                f"histograms/{annot_file.split('.csv')[0]}_histogram.png"
            )
            plot.figure.clf()


# This is horrible but for the alphabetical cell order it makes the colours match..
def _colour_bars(cell_colours):
    cell_colours[1], cell_colours[9] = cell_colours[9], cell_colours[1]
    cell_colours[2], cell_colours[9] = cell_colours[9], cell_colours[2]
    cell_colours[3], cell_colours[9] = cell_colours[9], cell_colours[3]
    cell_colours[4], cell_colours[10] = cell_colours[10], cell_colours[4]
    cell_colours[6], cell_colours[8] = cell_colours[8], cell_colours[6]
    cell_colours[7], cell_colours[9] = cell_colours[9], cell_colours[7]
    cell_colours[8], cell_colours[10] = cell_colours[10], cell_colours[8]
    cell_colours[9], cell_colours[10] = cell_colours[10], cell_colours[9]
    return cell_colours


if __name__ == "__main__":
    typer.run(main)
