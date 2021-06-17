import time
from pathlib import Path

import h5py
import umap
import umap.plot
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.neighbors import LocalOutlierFactor

import nucnet.db.eval_runs_interface as db
from nucnet.db.msfile_interface import get_msfile_by_run


def main():
    organ = "placenta"
    run_id = 5
    lab = "triin"
    slide_name = "9"

    print(f"Run id {run_id}, from lab {lab}, and slide {slide_name}")

    start = time.time()
    db.init()

    embeddings_dir = (
        Path(__file__).parent.parent.parent
        / "projects"
        / organ
        / "results"
        / "embeddings"
    )
    embeddings_path = db.get_embeddings_path(run_id, embeddings_dir)
    embeddings_file = embeddings_dir / embeddings_path

    with h5py.File(embeddings_file, "r") as f:
        subset_start = int(len(f["predictions"]) / 2)
        subset_end = subset_start + 50_000
        print(f"Running on {subset_end - subset_start} num of datapoints")
        predictions = f["predictions"][subset_start:subset_end]
        embeddings = f["embeddings"][subset_start:subset_end]
        coords = f["coords"][subset_start:subset_end]

    reducer = umap.UMAP(
        random_state=42,
        verbose=True,
        min_dist=0.1,
        n_neighbors=15,
        set_op_mix_ratio=0.25,
    )
    mapper = reducer.fit(embeddings)

    outlier_scores = LocalOutlierFactor(contamination=0.002).fit_predict(
        mapper.embedding_
    )

    outlying_coords = coords[outlier_scores == -1]
    outlying_preds = predictions[outlier_scores == -1]
    print(outlying_coords.shape)

    msfile = get_msfile_by_run(run_id)

    fig, axes = plt.subplots(7, 10, figsize=(10, 10))
    for i, ax in enumerate(axes.flatten()):
        cell_coords = outlying_coords[i]
        print(cell_coords)
        img = msfile.get_cell_tile_by_cell_coords(
            cell_coords[0], cell_coords[1], 200, 200
        )
        im = Image.fromarray(img.astype("uint8"))
        ax.imshow(im)
        plt.setp(ax, xticks=[], yticks=[])
    plt.tight_layout()

    project_root = str(embeddings_file).split("results")[0]
    vis_dir = (
        Path(project_root)
        / "visualisations"
        / "embeddings"
        / lab
        / f"slide_{slide_name}"
        / "outliers"
    )
    vis_dir.mkdir(parents=True, exist_ok=True)
    plot_name = f"start_{subset_start}_end_{subset_end}.png"
    print(f"saving plot to {vis_dir / plot_name}")

    plt.savefig(vis_dir / plot_name)

    end = time.time()
    duration = end - start
    print(f"total time: {duration:.4f} s")


if __name__ == "__main__":
    main()
