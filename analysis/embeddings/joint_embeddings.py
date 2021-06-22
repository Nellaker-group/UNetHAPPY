import time
from pathlib import Path

import typer
import umap
import umap.plot
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import happy.db.eval_runs_interface as db
from happy.hdf5.utils import get_embeddings_file, get_hdf5_datasets
from happy.cells.cells import get_organ


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
    start = time.time()

    organ = get_organ(organ_name)

    first_run = db.get_eval_run_by_id(first_run_id)
    first_slide_name = first_run.slide.slide_name
    first_lab_id = first_run.slide.lab

    second_run = db.get_eval_run_by_id(second_run_id)
    second_slide_name = second_run.slide.slide_name
    second_lab_id = second_run.slide.lab

    print(
        f"First: Run id {first_run_id}, from lab {first_lab_id}, "
        f"and slide {first_slide_name}"
    )
    print(
        f"Second: Run id {second_run_id}, from lab {second_lab_id}, "
        f"and slide {second_slide_name}"
    )

    # TODO: figure out the different colours here
    labels_dict = {
        0: "CYT",
        1: "FIB",
        2: "HOF",
        3: "SYN",
        4: "VEN",
        5: "CYT2",
        6: "FIB2",
        7: "HOF2",
        8: "SYN2",
        9: "VEN2",
    }
    colours_dict = {
        "CYT": "#24ff24",
        "FIB": "#fc0352",
        "HOF": "#ffff6d",
        "SYN": "#80b1d3",
        "VEN": "#fc8c44",
        "CYT2": "#0d8519",
        "FIB2": "#7b03fc",
        "HOF2": "#979903",
        "SYN2": "#0f0cad",
        "VEN2": "#731406",
    }

    first_embeddings_file = get_embeddings_file(project_name, first_run_id)
    second_embeddings_file = get_embeddings_file(project_name, second_run_id)
    first_predictions, first_embeddings, _, _, subset_end = get_hdf5_datasets(
        first_embeddings_file, subset_start, num_points
    )
    second_predictions, second_embeddings, _, _, subset_end = get_hdf5_datasets(
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
        project_root = Path(str(first_embeddings_file).split("results")[0])
        vis_dir = (
            project_root
            / "visualisations"
            / "embeddings"
            / "joint"
            / f"labs_{first_lab_id}_and_{second_lab_id}"
            / f"slides_{first_slide_name}_and_{second_slide_name}"
        )
        vis_dir.mkdir(parents=True, exist_ok=True)
        plot_name = f"start_{subset_start}_end_{subset_end}.png"
        print(f"saving plot to {vis_dir / plot_name}")
        plot = umap.plot.points(
            mapper, labels=both_predictions_labelled, color_key=colours_dict
        )
        plot.figure.savefig(f"{vis_dir / plot_name}")
    elif dimensions == 3:
        result = mapper.transform(both_embeddings)
        # TODO: figure out the different colours here
        custom_colours = np.array(
            [
                "#6cd4a4",
                "#ae0848",
                "#979903",
                "#80b1d3",
                "#fc8c44",
                "#24ff24",
                "#7b03fc",
                "#ffff6d",
                "#0f0cad",
                "#fc0352",
            ]
        )

        matplotlib.use("TkAgg")
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            result[:, 0],
            result[:, 1],
            result[:, 2],
            c=custom_colours[both_predictions],
            s=1,
        )
        plt.show()

    end = time.time()
    duration = end - start
    print(f"total time: {duration:.4f} s")


if __name__ == "__main__":
    typer.run(main)
