import pathlib
import time

import h5py
import umap
import umap.plot
import pandas as pd
import numpy as np
from bokeh.plotting import output_file, show, save

import nucnet.db.eval_runs_interface as db


def main():
    run_id = 16
    lab = "triin"
    slide_name = "139"
    mode = "interactive"

    print(f"Run id {run_id}, from lab {lab}, and slide {slide_name}")

    start = time.time()
    db.init()

    # embeddings_dir = "/well/nellaker/data/claudiav/nucnet_master/Results/embeddings/"
    embeddings_dir = "../Results/embeddings/"
    embeddings_path = db.get_embeddings_path(run_id, embeddings_dir)
    embeddings_file = f"{embeddings_dir}{embeddings_path}"

    with h5py.File(embeddings_file, "r") as f:
        subset_start = 0
        subset_end = subset_start + 50000
        print(f"Running on {subset_end - subset_start} num of datapoints")
        predictions = f["predictions"][subset_start:subset_end]
        embeddings = f["embeddings"][subset_start:subset_end]
        coords = f['coords'][subset_start:subset_end]
        confidence = f['confidence'][subset_start:subset_end]

    reducer = umap.UMAP(random_state=42, verbose=True, min_dist=0.1, n_neighbors=15)
    mapper = reducer.fit(embeddings)

    if mode == 'interactive':
        output_file(f'subset_slide_{slide_name}.html',
                    title=f"UMAP Embeddings of Slide {slide_name}")

        embeddings_save = embeddings_file.split("run")[0]
        plot_name = f"subset_slide_{slide_name}.html"
        print(f"saving interactive to {embeddings_save}/{plot_name}")

        label_colours = {
            0: "#6cd4a4", 1: "#ae0848", 2: "#f7f431", 3: "#80b1d3", 4: "#fc8c44"
        }

        df = pd.DataFrame({"pred": predictions, "confidence": confidence,
                           "x_": coords[:, 0], 'y_': coords[:, 1]})
        df['pred'] = df.pred.map({0: "CYT", 1: "FIB", 2: "HOF", 3: "SYN", 4: "VEN"})

        plot = umap.plot.interactive(mapper, labels=predictions,
                                     color_key=label_colours,
                                     interactive_text_search=True, hover_data=df,
                                     point_size=2)
        show(plot)
        save(plot, f"{embeddings_save}{plot_name}")
    else:
        project_root = embeddings_file.split("Results")[0]
        vis_dir = pathlib.Path(f"{project_root}Visualisations/"
                               f"embeddings/{lab}/slide_{slide_name}/")
        vis_dir.mkdir(parents=True, exist_ok=True)

        plot_name = f"start_{subset_start}_end_{subset_end}.png"
        print(f"saving plot to {vis_dir}/{plot_name}")

        labels_dict = {
            0: "CYT", 1: "FIB", 2: "HOF", 3: "SYN", 4: "VEN"
        }
        colours_dict = {
            "CYT": "#6cd4a4", "FIB": "#ae0848", "HOF": "#f7f431", "SYN": "#80b1d3",
            "VEN": "#fc8c44"
        }

        predictions_labelled = np.vectorize(labels_dict.get)(predictions)

        plot = umap.plot.points(mapper, labels=predictions_labelled,
                                color_key=colours_dict)
        plot.figure.savefig(f"{vis_dir}/{plot_name}")

    end = time.time()
    duration = (end - start)
    print(f"total time: {duration:.4f} s")


if __name__ == '__main__':
    main()
