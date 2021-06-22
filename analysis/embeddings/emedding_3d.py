import time

import typer
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import numpy as np

from happy.hdf5.utils import get_embeddings_file, get_hdf5_datasets
from happy.cells.cells import get_organ


def main(
    organ_name: str = typer.Option(...),
    project_name: str = typer.Option(...),
    run_id: int = typer.Option(...),
    subset_start: float = 0.0,
    num_points: int = 5000,
):
    """Visualise a 3d UMAP from the cell embedding vectors.

    Args:
        organ_name: name of the organ from which to get the cell types
        project_name: name of the project dir to save to
        run_id: id of the run which created the UMAP embeddings
        subset_start: at which index or proportion of the file to start (int or float)
        num_points: number of points to include in the UMAP from subset_start onwards
    """
    matplotlib.use("TkAgg")
    timer_start = time.time()
    organ = get_organ(organ_name)

    sns.set(style="white", context="poster", rc={"figure.figsize": (14, 10)})
    custom_colours = np.array([cell.colour for cell in organ.cell])

    embeddings_file = get_embeddings_file(project_name, run_id)
    predictions, embeddings, _, _, _, _ = get_hdf5_datasets(
        embeddings_file, subset_start, num_points
    )

    reducer = umap.UMAP(
        random_state=42, verbose=True, min_dist=0.1, n_neighbors=15, n_components=3
    )
    result = reducer.fit_transform(embeddings)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        result[:, 0], result[:, 1], result[:, 2], c=custom_colours[predictions], s=1
    )
    plt.show()

    timer_end = time.time()
    print(f"total time: {timer_end - timer_start:.4f} s")


if __name__ == "__main__":
    typer.run(main)
