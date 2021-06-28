import time
from pathlib import Path

import typer
import umap
import umap.plot
import matplotlib.pyplot as plt
import numpy as np

import happy.db.eval_runs_interface as db
from happy.hdf5.utils import get_embeddings_file, get_hdf5_datasets
from happy.cells.cells import get_organ
from utils import setup, embeddings_results_path
from plots import plot_3d


def main(
    organ_name: str = typer.Option(...),
    project_name: str = typer.Option(...),
    first_run_id: int = typer.Option(...),
    second_run_id: int = typer.Option(...),
    subset_start: float = 0.0,
    num_points: int = -1,
    dimensions: int = 2,
):
    """Plots and saves a UMAP from two cell embedding vectors of different slides.

    Can be 3d (with few points) or a 2d png.

    Args:
        organ_name: name of the organ from which to get the cell types
        project_name: name of the project dir to save to
        first_run_id: id of the first run which created the UMAP embeddings
        second_run_id: id of the second run which created the UMAP embeddings
        subset_start: at which index or proportion of the file to start (int or float)
        num_points: number of points to include in the UMAP from subset_start onwards
        dimensions: 2d or 3d UMAP plot
    """
    db.init()
    timer_start = time.time()

    first_lab_id, first_slide_name = setup(db, first_run_id)
    second_lab_id, second_slide_name = setup(db, second_run_id)
    organ = get_organ(organ_name)

    first_labels_dict = {cell.id: cell.label for cell in organ.cells}
    second_labels_dict = {cell.id + 5: cell.label + "2" for cell in organ.cells}
    labels_dict = {**first_labels_dict, **second_labels_dict}

    first_colours_dict = {cell.label: cell.colour for cell in organ.cells}
    second_colours_dict = {cell.label + "2": cell.alt_colour for cell in organ.cells}
    colours_dict = {**first_colours_dict, **second_colours_dict}

    first_embeddings_file = get_embeddings_file(project_name, first_run_id)
    second_embeddings_file = get_embeddings_file(project_name, second_run_id)
    first_predictions, first_embeddings, _, _, start, end = get_hdf5_datasets(
        first_embeddings_file, subset_start, num_points
    )
    second_predictions, second_embeddings, _, _, _, _ = get_hdf5_datasets(
        second_embeddings_file, subset_start, num_points
    )

    second_predictions = second_predictions + 5
    both_embeddings = np.append(first_embeddings, second_embeddings, axis=0)
    both_predictions = np.append(first_predictions, second_predictions, axis=0)

    both_predictions_labelled = np.vectorize(labels_dict.get)(both_predictions)

    reducer = umap.UMAP(
        random_state=42,
        verbose=True,
        min_dist=0.1,
        n_neighbors=15,
        n_components=dimensions,
    )
    mapper = reducer.fit(both_embeddings)

    if dimensions == 2:
        save_dir = embeddings_results_path(
            first_embeddings_file, first_lab_id, first_slide_name
        )
        save_dir = Path(*save_dir.parts[:-2])
        save_dir = (
            save_dir
            / "joint"
            / f"labs_{first_lab_id}_and_{second_lab_id}"
            / f"slides_{first_slide_name}_and_{second_slide_name}"
        )
        save_dir.mkdir(parents=True, exist_ok=True)

        plot_name = f"{start}-{end}.png"
        print(f"saving plot to {save_dir / plot_name}")
        plot = umap.plot.points(
            mapper, labels=both_predictions_labelled, color_key=colours_dict
        )
        plot.figure.savefig(save_dir / plot_name)
    elif dimensions == 3:
        result = mapper.transform(both_embeddings)
        first_colours = np.array([cell.colour for cell in organ.cell])
        second_colours = np.array([cell.alt_colour for cell in organ.cell])
        custom_colours = np.append(first_colours, second_colours, axis=0)

        plot_3d(organ, result, both_predictions, custom_colours)
        plt.show()

    timer_end = time.time()
    print(f"total time: {timer_end - timer_start:.4f} s")


if __name__ == "__main__":
    typer.run(main)
