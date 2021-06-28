from pathlib import Path
import pandas as pd
import typer

import happy.db.eval_runs_interface as db
from happy.hdf5.utils import filter_hdf5
from happy.cells.cells import get_organ
from happy.hdf5.utils import get_embeddings_file


def main(
    organ_name: str = typer.Option(...),
    project_name: str = typer.Option(...),
    run_id: int = typer.Option(...),
    slide_name: str = typer.Option(...),
    nuclei_only: bool = False,
    filtered: bool = False,
    min_conf: float = 0.0,
    max_conf: float = 1.0,
):
    """Saves model predictions from database into tsvs that QuPath can read

    Args:
        organ_name: name of the organ from which to get the cell types
        project_name: name of the project directory
        run_id: id of the run which generated the predictions
        slide_name: shorthand name of the slide for naming the tsv file
        nuclei_only: a flag for when there are no cell predictions in the db
        filtered: whether to filter by network confidence
        min_conf: min network confidence to include
        max_conf: max network confidence to include
    """
    db.init()

    projects_dir = Path(__file__).parent.parent / "projects"
    save_dir = projects_dir / project_name / "results" / "tsvs"
    organ = get_organ(organ_name)

    if not filtered:
        save_path = save_dir / f"{slide_name}.tsv"
        coords, preds = _get_db_predictions(run_id)
    else:
        save_path = save_dir / f"{min_conf}_{max_conf}_{slide_name}.tsv"
        coords, preds = _get_filtered_confidence_predictions(
            organ, project_name, run_id, min_conf, max_conf
        )

    coord_to_tsv(coords, preds, save_path, organ, nuclei_only)


def coord_to_tsv(coords, preds, save_path, organ, nuclei_only=False):
    xs = [coord["x"] for coord in coords]
    ys = [coord["y"] for coord in coords]
    if not nuclei_only:
        cell_class = [
            organ.cells[pred["cell_class"]].label if pred is not None else "NA"
            for pred in preds
        ]
    else:
        cell_class = ["Nuclei" for _ in coords]

    print(f"saving {len(preds)} cells to tsv")
    df = pd.DataFrame({"x": xs, "y": ys, "class": cell_class})
    df.to_csv(save_path, sep="\t", index=False)


def _get_db_predictions(run_id):
    coords = db.get_predictions(run_id)
    return coords, coords


def _get_filtered_confidence_predictions(
    organ, project_name, run_id, metric_start, metric_end
):
    embeddings_file = get_embeddings_file(project_name, run_id)

    predictions, _, coords, _, _, _, _ = filter_hdf5(
        organ,
        embeddings_file,
        start=0,
        num_points=-1,
        metric_type="confidence",
        metric_start=metric_start,
        metric_end=metric_end,
    )
    print(f"confidence bounds: {metric_start}-{metric_end}")

    return coords, predictions


if __name__ == "__main__":
    typer.run(main)
