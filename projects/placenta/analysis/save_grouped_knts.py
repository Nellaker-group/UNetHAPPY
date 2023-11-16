from typing import Optional

import typer
import numpy as np
import pandas as pd

from happy.organs import get_organ
import happy.db.eval_runs_interface as db
from projects.placenta.graphs.processing.process_knots import process_knt_cells
from happy.graph.graph_creation.get_and_process import get_groundtruth_patch
from happy.graph.graph_creation.get_and_process import get_hdf5_data
from happy.utils.utils import get_project_dir
import h5py


def main(
    run_id: int = typer.Option(...),
    project_name: str = "placenta",
    x_min: int = 0,
    y_min: int = 0,
    width: int = -1,
    height: int = -1,
    make_tsv: bool = False,
    tissue_label_tsv: Optional[str] = None,
):
    # Create database connection
    db.init()
    project_dir = get_project_dir(project_name)
    organ = get_organ("placenta")
    cell_label_mapping = {cell.id: cell.label for cell in organ.cells}

    # Get hdf5 datasets contained in specified box/patch of WSI
    hdf5_data = get_hdf5_data(
        project_name, run_id, x_min, y_min, width, height
    )

    if tissue_label_tsv:
        px, py, tissues = get_groundtruth_patch(
            organ, project_dir, x_min, y_min, width, height, tissue_label_tsv
        )
    else:
        tissues = None

    # Turn isolated knts into syn and group large knts into one point
    grouped_hdf5_data, inds_to_remove = process_knt_cells(
        hdf5_data, organ, 50, 3
    )
    predictions = grouped_hdf5_data.cell_predictions
    embeddings = grouped_hdf5_data.cell_embeddings
    coords = grouped_hdf5_data.coords
    confidence = grouped_hdf5_data.cell_confidence

    # Remove points from tissue ground truth as well
    if tissues is not None and len(inds_to_remove) > 0:
        tissues = np.delete(tissues, inds_to_remove, axis=0)
        px = np.delete(px, inds_to_remove, axis=0)
        py = np.delete(py, inds_to_remove, axis=0)

        print(f"Making gt tissue file...")
        tissue_label_mapping = {tissue.id: tissue.label for tissue in organ.tissues}
        tissue_df = pd.DataFrame({"px": px, "py": py, "class": tissues})
        tissue_df["class"] = tissue_df["class"].map(tissue_label_mapping)
        tissue_df.to_csv(
            "plots/grouped_tissue.tsv", sep="\t", encoding="utf-8", index=False
        )

    if make_tsv:
        print(f"Making tsv file...")
        label_predictions = [cell_label_mapping[x] for x in predictions]
        df = pd.DataFrame(
            {"x": coords[:, 0], "y": coords[:, 1], "class": label_predictions}
        )
        df.to_csv("plots/grouped_knt_cells.tsv", sep="\t", index=False)

        print(f"Making h5py file...")
        new_h5_path = f"plots/run_{run_id}"
        with h5py.File(new_h5_path, "w") as f:
            f.create_dataset("predictions", data=predictions, dtype="int8")
            f.create_dataset("embeddings", data=embeddings, dtype="float32")
            f.create_dataset("confidence", data=confidence, dtype="float16")
            f.create_dataset("coords", data=coords, dtype="uint32")


if __name__ == "__main__":
    typer.run(main)
