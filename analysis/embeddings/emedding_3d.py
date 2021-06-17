import time

import h5py
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import numpy as np

import nucnet.db.eval_runs_interface as db


def main():
    matplotlib.use("TkAgg")
    start = time.time()

    db.init()
    sns.set(style='white', context='poster', rc={'figure.figsize': (14, 10)})
    custom_colours = np.array(["#6cd4a4", "#ae0848", "#f7f431", "#80b1d3", "#fc8c44"])

    run_id = 3
    embeddings_dir = "../../Results/embeddings/"
    embeddings_path = db.get_embeddings_path(run_id, embeddings_dir)
    embeddings_file = f"{embeddings_dir}{embeddings_path}"

    embeddings_save = embeddings_file.split("run")[0]
    plot_name = "subset1_slide_3.png"
    print(f"saving to {embeddings_save}{plot_name}")

    subset_start = 500_000
    subset_end = 550_000
    with h5py.File(embeddings_file, "r") as f:
        predictions = f["predictions"][subset_start:subset_end]
        embeddings = f["embeddings"][subset_start:subset_end]

    reducer = umap.UMAP(random_state=42, verbose=True, min_dist=0.1,
                        n_neighbors=15, n_components=3)
    result = reducer.fit_transform(embeddings)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(result[:, 0], result[:, 1], result[:, 2], c=custom_colours[predictions],
               s=1)
    plt.show()

    end = time.time()
    duration = (end - start)
    print(f"total time: {duration:.4f} s")

if __name__ == '__main__':
    main()
