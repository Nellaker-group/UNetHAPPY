import typer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from happy.organs.organs import get_organ
from happy.utils.utils import get_project_dir
import happy.db.eval_runs_interface as db
from happy.hdf5.utils import get_datasets_in_patch, get_embeddings_file
from knot_nuclei_to_point import process_knt_cells


def main(
    run_id: int = typer.Option(...),
    project_name: str = "placenta",
    exp_name: str = typer.Option(...),
    model_weights_dir: str = typer.Option(...),
    model_name: str = typer.Option(...),
    model_type: str = "sup_clustergcn",
    x_min: int = 0,
    y_min: int = 0,
    width: int = -1,
    height: int = -1,
    group_knts: bool = False,
):
    # Create database connection
    db.init()
    organ = get_organ("placenta")
    cell_label_mapping = {cell.id: cell.label for cell in organ.cells}
    project_dir = get_project_dir(project_name)

    # Get path to embeddings hdf5 files
    embeddings_path = get_embeddings_file(project_name, run_id)
    # Get hdf5 datasets contained in specified box/patch of WSI
    predictions, embeddings, coords, confidence = get_datasets_in_patch(
        embeddings_path, x_min, y_min, width, height
    )

    if group_knts:
        predictions, embeddings, coords, confidence = process_knt_cells(
            predictions,
            embeddings,
            coords,
            confidence,
            organ,
            100,
            3,
            width,
            height,
            plot=False,
        )

    # print cell predictions from hdf5 file
    unique_cells, cell_counts = np.unique(predictions, return_counts=True)
    unique_cell_labels = []
    for label in unique_cells:
        unique_cell_labels.append(cell_label_mapping[label])
    unique_cell_counts = dict(zip(unique_cell_labels, cell_counts))
    print(f"Num cell predictions per label: {unique_cell_counts}")

    # print tissue predictions from tsv file
    pretrained_path = (
        project_dir
        / "results"
        / "graph"
        / model_type
        / exp_name
        / model_weights_dir
        / "eval"
        / model_name
    )
    tissue_preds = pd.read_csv(pretrained_path / "tissue_preds.tsv", sep="\t")
    unique_tissues, tissue_counts = np.unique(tissue_preds["class"], return_counts=True)
    unique_tissue_counts = dict(zip(unique_tissues, tissue_counts))
    print(f"Num tissue predictions per label: {unique_tissue_counts}")

    # get number of cell types within each tissue type
    cell_predictions = [cell_label_mapping[cell] for cell in predictions]
    get_cells_within_tissues(coords, cell_predictions, tissue_preds["class"])


# find cell types within each tissue type and plot as stacked bar chart
def get_cells_within_tissues(coords, cell_predictions, tissue_predictions):
    combined_df = pd.DataFrame(
        {
            "x": coords[:, 0],
            "y": coords[:, 1],
            "Cells": cell_predictions,
            "Tissues": tissue_predictions,
        }
    )
    grouped_df = (
        combined_df.groupby(["Tissues", "Cells"]).size().reset_index(name="count")
    )
    grouped_df["prop"] = grouped_df.groupby(["Tissues"])["count"].transform(
        lambda x: x * 100 / x.sum()
    )
    prop_df = grouped_df.pivot_table(index="Tissues", columns="Cells", values="prop")
    prop_df.plot(kind="bar", stacked=True)
    plt.show()


if __name__ == "__main__":
    typer.run(main)
