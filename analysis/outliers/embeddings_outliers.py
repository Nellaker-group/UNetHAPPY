import time
from pathlib import Path

import typer
import umap
import umap.plot
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.neighbors import LocalOutlierFactor

import happy.db.eval_runs_interface as db
from happy.db.msfile_interface import get_msfile_by_run
from happy.hdf5.utils import get_embeddings_file, get_hdf5_datasets
from happy.organs.organs import get_organ


def main(
    project_name: str = typer.Option(...),
    organ_name: str = typer.Option(...),
    run_id: int = typer.Option(...),
    subset_start: float = 0.0,
    num_points: int = -1,
    plot_embedding: bool = False,
):
    """Finds the top 100 outlier points from the UMAP embeddings

    Saves the top 100 outliers as 200x200 images with the points in the centre.
    Prints the coordinates, class prediction, and network confidence of those outliers.
    Plot displays the class prediction and network confidence per image.

    Args:
        project_name: name of the project dir to save to
        organ_name: name of the organ to get the cell colours
        run_id: id of the run which created the UMAP embeddings
        subset_start: at which index or proportion of the file to start (int or float)
        num_points: number of points to include in the UMAP from subset_start onwards
        plot_embedding: plot the embedding from which the outliers were generated
    """
    db.init()
    organ = get_organ(organ_name)
    cell_mapping = {cell.id: cell.label for cell in organ.cells}
    timer_start = time.time()

    run = db.get_eval_run_by_id(run_id)
    slide_name = run.slide.slide_name
    lab_id = run.slide.lab
    print(f"Run id {run_id}, from lab {lab_id}, and slide {slide_name}")

    embeddings_file = get_embeddings_file(project_name, run_id)
    predictions, embeddings, coords, conf, start, end = get_hdf5_datasets(
        embeddings_file, subset_start, num_points
    )

    # Generate UMAP
    reducer = umap.UMAP(
        random_state=42,
        verbose=True,
        min_dist=0.1,
        n_neighbors=15,
        set_op_mix_ratio=0.25,
    )
    mapper = reducer.fit(embeddings)

    # Get proportion of contamination for 100 outliers
    total_points = end if num_points == -1 else num_points
    proportion = 100 / total_points

    # Find outliers
    outlier_scores = LocalOutlierFactor(contamination=proportion).fit_predict(
        mapper.embedding_
    )
    outlying_coords = coords[outlier_scores == -1]
    outlying_preds = predictions[outlier_scores == -1]
    outlying_confidence = conf[outlier_scores == -1]
    # Sort outliers by network confidence
    sorted_indicies = np.argsort(outlying_confidence)
    outlying_confidence = np.sort(outlying_confidence)
    outlying_coords = outlying_coords[sorted_indicies[::-1]]
    outlying_preds = outlying_preds[sorted_indicies[::-1]]

    msfile = get_msfile_by_run(run_id)

    fig, axes = plt.subplots(10, 10, figsize=(10, 10))
    for i, ax in enumerate(axes.flatten()):
        cell_coords = outlying_coords[i]
        cell_preds = outlying_preds[i]
        cell_conf = outlying_confidence[i]
        print(
            f"x{cell_coords[0]}, y{cell_coords[1]}, "
            f"{cell_mapping[cell_preds]}, {cell_conf:.2f}"
        )
        img = msfile.get_cell_tile_by_cell_coords(
            cell_coords[0], cell_coords[1], 200, 200
        )
        im = Image.fromarray(img.astype("uint8"))
        ax.imshow(im)
        ax.set_title(f"{cell_mapping[cell_preds]} {cell_conf:.2f}", fontsize=10)
        plt.setp(ax, xticks=[], yticks=[])
    plt.tight_layout()

    project_root = Path(str(embeddings_file).split("results")[0])
    vis_dir = (
        project_root
        / "visualisations"
        / "embeddings"
        / f"lab_{lab_id}"
        / f"slide_{slide_name}"
        / "outliers"
    )
    vis_dir.mkdir(parents=True, exist_ok=True)
    plot_name = f"{start}-{end}.png"
    print(f"saving plot to {vis_dir / plot_name}")
    plt.savefig(vis_dir / plot_name)

    if plot_embedding:
        save_dir = Path(*vis_dir.parts[:-1])
        plot_name = f"{start}-{end}.png"
        colours_dict = {cell.label: cell.colour for cell in organ.cells}
        predictions_labelled = np.array(
            [organ.cells[pred].label for pred in predictions]
        )
        plot = umap.plot.points(
            mapper, labels=predictions_labelled, color_key=colours_dict
        )
        print(f"saving plot to {save_dir / plot_name}")
        plot.figure.savefig(save_dir / plot_name)

    timer_end = time.time()
    print(f"total time: {timer_end - timer_start:.4f} s")


if __name__ == "__main__":
    typer.run(main)
