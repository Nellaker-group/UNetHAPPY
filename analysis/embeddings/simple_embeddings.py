import time

import typer
import matplotlib.pyplot as plt
import seaborn as sns
import umap

import happy.db.eval_runs_interface as db
from happy.hdf5.utils import get_embeddings_file, get_hdf5_datasets
from happy.cells.cells import get_organ
from utils import embeddings_results_path, setup


def main(
    organ_name: str = typer.Option(...),
    project_name: str = typer.Option(...),
    run_id: int = typer.Option(...),
    subset_start: float = 0.0,
    num_points: int = 5000,
):
    """Uses seaborn to make a simple, small plot of the UMAP embedding vectors

    Args:
        organ_name: name of the organ from which to get the cell types
        project_name: name of the project dir to save to
        run_id: id of the run which created the UMAP embeddings
        subset_start: at which index or proportion of the file to start (int or float)
        num_points: number of points to include in the UMAP from subset_start onwards
    """
    db.init()
    timer_start = time.time()

    lab_id, slide_name = setup(db, run_id)
    organ = get_organ(organ_name)

    sns.set(style='white', context='poster', rc={'figure.figsize': (14, 10)})
    custom_colours = [cell.colour for cell in organ.cells]

    embeddings_file = get_embeddings_file(project_name, run_id)
    (predictions, embeddings, coords, confidence, start, end) = get_hdf5_datasets(
        embeddings_file, subset_start, num_points
    )

    reducer = umap.UMAP(random_state=42, verbose=True, min_dist=0.0)
    result = reducer.fit_transform(embeddings)

    save_dir = embeddings_results_path(embeddings_file, lab_id, slide_name)
    save_path = save_dir / f"{start}-{end}.png"

    ax = sns.scatterplot(result[:, 0], result[:, 1], hue=predictions, s=10,
                         palette=sns.color_palette(custom_colours))
    legend = ax.legend_
    for t, l in zip(legend.texts, tuple([cell.label for cell in organ.cells])):
        t.set_text(l)
    ax.set_title('UMAP projection of cell classes')
    print(f"saving to {save_path}")
    plt.savefig(save_path)

    timer_end = time.time()
    print(f"total time: {timer_end - timer_start:.4f} s")

if __name__ == '__main__':
    typer.run(main)
