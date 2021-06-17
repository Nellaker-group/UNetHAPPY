import time

import h5py
import matplotlib.pyplot as plt
import seaborn as sns
import umap

import happy.db.eval_runs_interface as db


def main():
    start = time.time()

    db.init()
    sns.set(style='white', context='poster', rc={'figure.figsize': (14, 10)})
    custom_colours = ["#6cd4a4", "#ae0848", "#f7f431", "#80b1d3", "#fc8c44"]

    run_id = 1
    # TODO: change this path
    embeddings_dir = "../../Results/embeddings/"
    embeddings_path = db.get_embeddings_path(run_id, embeddings_dir)
    embeddings_file = f"{embeddings_dir}{embeddings_path}"

    embeddings_save = embeddings_file.split("run")[0]
    plot_name = "subset1_slide_3.png"
    print(f"saving to {embeddings_save}{plot_name}")

    subset_start = 0
    subset_end = 500_000
    with h5py.File(embeddings_file, "r") as f:
        predictions = f["predictions"][subset_start:subset_end]
        embeddings = f["embeddings"][subset_start:subset_end]

    reducer = umap.UMAP(random_state=42, verbose=True, min_dist=0.0)
    result = reducer.fit_transform(embeddings)

    ax = sns.scatterplot(result[:, 0], result[:, 1], hue=predictions, s=10,
                         palette=sns.color_palette(custom_colours))
    legend = ax.legend_
    for t, l in zip(legend.texts, ("CYT", "FIB", "HOF", "SYN", "VEN")):
        t.set_text(l)
    ax.set_title('UMAP projection of cell classes in triin slide 3')
    plt.savefig(f"{embeddings_save}{plot_name}")

    end = time.time()
    duration = (end - start)
    print(f"total time: {duration:.4f} s")

if __name__ == '__main__':
    main()
