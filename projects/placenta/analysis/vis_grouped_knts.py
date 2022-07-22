import typer
import pandas as pd

from happy.organs.organs import get_organ
import happy.db.eval_runs_interface as db
from projects.placenta.graphs.analysis.knot_nuclei_to_point import process_knt_cells
from projects.placenta.graphs.analysis.vis_graph_patch import visualize_points
from happy.hdf5.utils import (
    get_datasets_in_patch,
    get_embeddings_file,
    filter_by_cell_type,
)


def main(
    run_id: int = typer.Option(...),
    project_name: str = "placenta",
    x_min: int = 0,
    y_min: int = 0,
    width: int = -1,
    height: int = -1,
    radius: int = 50,
    cut_off_count: int = 3,
    make_tsv: bool = False,
):
    # Create database connection
    db.init()
    organ = get_organ("placenta")
    cell_label_mapping = {cell.id: cell.label for cell in organ.cells}

    # Get path to embeddings hdf5 files
    embeddings_path = get_embeddings_file(project_name, run_id)
    # Get hdf5 datasets contained in specified box/patch of WSI
    predictions, embeddings, coords, confidence = get_datasets_in_patch(
        embeddings_path, x_min, y_min, width, height
    )

    predictions, embeddings, coords, confidence, _ = process_knt_cells(
        predictions,
        embeddings,
        coords,
        confidence,
        organ,
        radius,
        cut_off_count,
        plot=True,
    )

    predictions, embeddings, coords, confidence = filter_by_cell_type(
        predictions, embeddings, coords, confidence, "knt", organ
    )

    visualize_points(
        organ,
        "plots/grouped_KNT.png",
        coords,
        labels=predictions,
        width=width,
        height=height,
    )

    if make_tsv:
        print(f"Making tsv file...")
        label_predictions = [cell_label_mapping[x] for x in predictions]
        df = pd.DataFrame(
            {"x": coords[:, 0], "y": coords[:, 1], "class": label_predictions}
        )
        df.to_csv("plots/grouped_knt_cells.tsv", sep="\t", index=False)


if __name__ == "__main__":
    typer.run(main)
