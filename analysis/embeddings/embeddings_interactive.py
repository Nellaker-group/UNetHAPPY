from pathlib import Path
import time

import typer
import umap
import umap.plot
import pandas as pd
import numpy as np
from bokeh.plotting import output_file, show, save

import happy.db.eval_runs_interface as db
from happy.hdf5.utils import get_embeddings_file, get_hdf5_datasets
from happy.cells.cells import get_organ


def main(
    organ_name: str = typer.Option(...),
    project_name: str = typer.Option(...),
    run_id: int = typer.Option(...),
    subset_start: float = 0.0,
    num_points: int = -1,
    interactive: bool = False,
):
    """Plots and saves a UMAP from the cell emebdding vectors.

    Can be an interactive html file (open with browser or IDE) or simply a png.

    Args:
        organ_name: name of the organ from which to get the cell types
        project_name: name of the project dir to save to
        run_id: id of the run which created the UMAP embeddings
        subset_start: at which index of proportion of the file to start (int or float)
        num_points: number of points to include in the UMAP from subset_start onwards
        interactive: to save an interactive html or a png
    """
    db.init()
    start = time.time()

    run = db.get_eval_run_by_id(run_id)
    slide_name = run.slide.slide_name
    lab_id = run.slide.lab
    print(f"Run id {run_id}, from lab {lab_id}, and slide {slide_name}")
    organ = get_organ(organ_name)

    embeddings_file = get_embeddings_file(project_name, run_id)
    predictions, embeddings, coords, confidence, subset_end = get_hdf5_datasets(
        embeddings_file, subset_start, num_points
    )

    # Generate UMAP
    reducer = umap.UMAP(random_state=42, verbose=True, min_dist=0.1, n_neighbors=15)
    mapper = reducer.fit(embeddings)

    project_root = Path(str(embeddings_file).split("results")[0])
    vis_dir = (
        Path(project_root)
        / "visualisations"
        / "embeddings"
        / f"lab_{lab_id}"
        / f"slide_{slide_name}"
    )
    vis_dir.mkdir(parents=True, exist_ok=True)

    if interactive:
        plot_name = f"start_{subset_start}_end_{subset_end}.html"
        output_file(plot_name, title=f"UMAP Embeddings of Slide {slide_name}")

        label_colours = {cell.id: cell.colour for cell in organ.cells}
        label_ids = {cell.id: cell.label for cell in organ.cells}

        df = pd.DataFrame(
            {
                "pred": predictions,
                "confidence": confidence,
                "x_": coords[:, 0],
                "y_": coords[:, 1],
            }
        )
        df["pred"] = df.pred.map(label_ids)

        plot = umap.plot.interactive(
            mapper,
            labels=predictions,
            color_key=label_colours,
            interactive_text_search=True,
            hover_data=df,
            point_size=2,
        )
        show(plot)
        print(f"saving plot to {vis_dir / plot_name}")
        save(plot, vis_dir / plot_name)
    else:
        plot_name = f"start_{subset_start}_end_{subset_end}.png"

        colours_dict = {cell.label: cell.colour for cell in organ.cells}
        predictions_labelled = np.array(
            [organ.cells[pred].label for pred in predictions]
        )

        plot = umap.plot.points(
            mapper, labels=predictions_labelled, color_key=colours_dict
        )
        print(f"saving plot to {vis_dir / plot_name}")
        plot.figure.savefig(f"{vis_dir / plot_name}")

    end = time.time()
    duration = end - start
    print(f"total time: {duration:.4f} s")


if __name__ == "__main__":
    typer.run(main)
