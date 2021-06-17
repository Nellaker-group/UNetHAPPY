import time

import h5py
import umap
import umap.plot
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import happy.db.eval_runs_interface as db


def main():
    dimensions = 2

    first_run_id = 3
    first_lab = "triin"
    first_slide_name = "3"

    second_run_id = 4
    second_lab = "hagit"
    second_slide_name = "1003608"

    print(f"First: Run id {first_run_id}, from lab {first_lab}, "
          f"and slide {first_slide_name}")
    print(f"Second: Run id {second_run_id}, from lab {second_lab}, "
          f"and slide {second_slide_name}")

    start = time.time()
    db.init()
    labels_dict = {
        0: "CYT", 1: "FIB", 2: "HOF", 3: "SYN", 4: "VEN",
        5: "CYT2", 6: "FIB2", 7: "HOF2", 8: "SYN2", 9: "VEN2"
    }
    colours_dict = {
        "CYT": "#24ff24", "FIB": "#fc0352", "HOF": "#ffff6d", "SYN": "#80b1d3",
        "VEN": "#fc8c44", "CYT2": "#0d8519", "FIB2": "#7b03fc", "HOF2": "#979903",
        "SYN2": "#0f0cad", "VEN2": "#731406"
    }

    embeddings_dir = "../../Results/embeddings/"
    first_embeddings_path = db.get_embeddings_path(first_run_id, embeddings_dir)
    first_embeddings_file = f"{embeddings_dir}{first_embeddings_path}"

    second_embeddings_path = db.get_embeddings_path(second_run_id, embeddings_dir)
    second_embeddings_file = f"{embeddings_dir}{second_embeddings_path}"

    with h5py.File(first_embeddings_file, "r") as f:
        subset_start = 200_000
        subset_end = 230_000
        print(f"First on {subset_end - subset_start} num of datapoints")
        first_predictions = f["predictions"][subset_start:subset_end]
        first_embeddings = f["embeddings"][subset_start:subset_end]

    with h5py.File(second_embeddings_file, "r") as f:
        subset_start = 200_000
        subset_end = 230_000
        print(f"Second on {subset_end - subset_start} num of datapoints each")
        second_predictions = f["predictions"][subset_start:subset_end]
        second_embeddings = f["embeddings"][subset_start:subset_end]

    second_predictions = second_predictions + 5
    both_embeddings = np.append(first_embeddings, second_embeddings, axis=0)
    both_predictions = np.append(first_predictions, second_predictions, axis=0)

    both_predictions_labelled = np.vectorize(labels_dict.get)(both_predictions)

    reducer = umap.UMAP(random_state=42, verbose=True, min_dist=0.1, n_neighbors=15,
                        n_components=dimensions)
    mapper = reducer.fit(both_embeddings)

    if dimensions == 2:
        project_root = first_embeddings_file.split("Results")[0]
        vis_dir = f"{project_root}Visualisations/embeddings" \
                  f"/joint/slide_{first_slide_name}_and_{second_slide_name}/"
        plot_name = f"start_{subset_start}_end_{subset_end}.png"
        print(f"saving plot to {vis_dir}{plot_name}")
        plot = umap.plot.points(mapper, labels=both_predictions_labelled,
                                color_key=colours_dict)
        plot.figure.savefig(f"{vis_dir}{plot_name}")
    elif dimensions == 3:
        result = mapper.transform(both_embeddings)
        custom_colours = np.array(
            ["#6cd4a4", "#ae0848", "#979903", "#80b1d3", "#fc8c44",
             "#24ff24", "#7b03fc", "#ffff6d", "#0f0cad", "#fc0352"])

        matplotlib.use("TkAgg")
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(result[:, 0], result[:, 1], result[:, 2],
                   c=custom_colours[both_predictions], s=1)
        plt.show()

    end = time.time()
    duration = (end - start)
    print(f"total time: {duration:.4f} s")


if __name__ == '__main__':
    main()
