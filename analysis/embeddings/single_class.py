import time

import typer
import umap
import umap.plot
from bokeh.plotting import show, save

import happy.db.eval_runs_interface as db
from happy.hdf5 import filter_hdf5, get_embeddings_file
from happy.organs import get_organ
from utils import embeddings_results_path, setup
from plots import plot_interactive, plot_cell_umap


def main(
    organ_name: str = typer.Option(...),
    project_name: str = typer.Option(...),
    run_id: int = typer.Option(...),
    subset_start: float = 0.0,
    num_points: int = 5000,
    interactive: bool = False,
    cell_class: str = typer.Option(...),
):
    """Plots and saves a UMAP from the embedding vectors of a single cell.

    Can be an interactive html file (open with browser or IDE) or simply a png.

    Args:
        organ_name: name of the organ from which to get the cell types
        project_name: name of the project dir to save to
        run_id: id of the run which created the UMAP embeddings
        subset_start: at which index or proportion of the file to start (int or float)
        num_points: number of points to include in the UMAP from subset_start onwards
        interactive: to save an interactive html or a png
        cell_class: a string label of the cell class to filter for
    """
    db.init()
    timer_start = time.time()

    lab_id, slide_name = setup(db, run_id)
    organ = get_organ(organ_name)
    print(f"Filtering by class: {cell_class}")

    embeddings_file = get_embeddings_file(project_name, run_id)
    predictions, embeddings, coords, confidence, start, end, num_filtered = filter_hdf5(
        organ,
        embeddings_file,
        start=subset_start,
        num_points=num_points,
        metric_type="cell_class",
        metric_start=cell_class,
    )

    reducer = umap.UMAP(random_state=42, verbose=True, min_dist=0.1, n_neighbors=15)
    mapper = reducer.fit(embeddings)

    save_dir = embeddings_results_path(embeddings_file, lab_id, slide_name)
    save_dir = save_dir / "single_cell"
    save_dir.mkdir(exist_ok=True)

    if interactive:
        plot_name = f"{start}-{end}_{cell_class}.html"
        plot = plot_interactive(
            plot_name, slide_name, organ, predictions, confidence, coords, mapper
        )
        show(plot)
        print(f"saving interactive to {save_dir / plot_name}")
        save(plot, save_dir / plot_name)
    else:
        plot_name = f"{start}-{end}_{cell_class}.png"
        plot = plot_cell_umap(organ, predictions, mapper)
        print(f"saving plot to {save_dir / plot_name}")
        plot.figure.savefig(save_dir / plot_name)

    timer_end = time.time()
    print(f"total time: {timer_end - timer_start:.4f} s")


if __name__ == "__main__":
    typer.run(main)
