import pathlib
import time

import umap
import umap.plot
import pandas as pd
import numpy as np
from bokeh.plotting import output_file, show, save

import happy.db.eval_runs_interface as db
from happy.hdf5.utils import filter_hdf5


def main():
    run_id = 16
    lab = "triin"
    slide_name = "139"
    mode = "points"
    confidence_start = 0.9
    confidence_end = 1.0

    subset_start = 0
    num_points = -1

    print(f"Run id {run_id}, from lab {lab}, and slide {slide_name}")
    print(f"Filtering by confidence ranges: {confidence_start}-{confidence_end}")

    start = time.time()
    db.init()

    embeddings_dir = "../../Results/embeddings/"
    embeddings_path = db.get_embeddings_path(run_id, embeddings_dir)
    embeddings_file = f"{embeddings_dir}{embeddings_path}"

    predictions, embeddings, coords, confidence, subset_end, num_filtered = \
        filter_hdf5(embeddings_file, start=subset_start, num_points=num_points,
                    metric_type="confidence", metric_start=confidence_start,
                    metric_end=confidence_end)

    reducer = umap.UMAP(random_state=42, verbose=True, min_dist=0.1, n_neighbors=15)
    mapper = reducer.fit(embeddings)

    if mode == 'interactive':
        output_file(f'subset_slide_{slide_name}.html',
                    title=f"UMAP Embeddings of Slide {slide_name}")

        embeddings_save = embeddings_file.split("run")[0]
        plot_name = f"subset_slide_{slide_name}_confidence_" \
                    f"{confidence_start}-{confidence_end}.html"
        print(f"saving interactive to {embeddings_save}interactive/{plot_name}")

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
                               f"embeddings/{lab}/slide_{slide_name}/confidence/")
        vis_dir.mkdir(parents=True, exist_ok=True)

        plot_name = f"{confidence_start}-{confidence_end}_start_" \
                    f"{subset_start}_end_{subset_end}_num_{num_filtered}.png"
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
