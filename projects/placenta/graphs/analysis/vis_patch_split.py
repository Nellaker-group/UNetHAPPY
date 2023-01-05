from typing import List
import time

import typer
import numpy as np
import pandas as pd

from happy.graph.create_graph import get_groundtruth_patch
from projects.placenta.graphs.analysis.vis_graph_patch import visualize_points
from happy.graph.create_graph import get_nodes_within_tiles
from happy.utils.utils import get_project_dir
from happy.organs import get_organ


def main(
    project_name: str = "placenta",
    organ_name: str = "placenta",
    dataset_patch_files: List[str] = typer.Option([]),
    tissue_label_tsv: str = "96_tissue_points.tsv",
    slide_name: str = "slide_96-2019-09-05 22.47.40.ndpi",
):
    project_dir = get_project_dir(project_name)
    organ = get_organ(organ_name)

    xs, ys, tissue_class = get_groundtruth_patch(
        organ, project_dir, 0, 0, -1, -1, tissue_label_tsv
    )
    tissue_class = np.zeros(tissue_class.shape)

    timer_start = time.time()
    print("Creating groundtruth patches from files")
    for file in dataset_patch_files:
        patches_df = pd.read_csv(project_dir / "graph_splits" / file)
        for row in patches_df.itertuples(index=False):
            patch_node_inds = get_nodes_within_tiles(
                (row.x, row.y), row.width, row.height, xs, ys
            )
            patch_xs, patch_ys, patch_tissue_class = get_groundtruth_patch(
                organ,
                project_dir,
                row.x,
                row.y,
                row.width,
                row.height,
                tissue_label_tsv,
            )
            tissue_class[patch_node_inds] = patch_tissue_class
    timer_end = time.time()
    print(f"total time: {timer_end - timer_start:.4f} s")

    save_dir = (
        project_dir / "visualisations" / "graphs" / "lab_1" / slide_name / "patches_gt"
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    plot_name = f"patches.png"
    save_path = save_dir / plot_name

    colours_dict = {tissue.id: tissue.colour for tissue in organ.tissues}
    colours = [colours_dict[label] for label in tissue_class]
    print("Visualising patches")
    visualize_points(
        organ,
        save_path,
        np.stack((xs, ys), axis=1),
        colours=colours,
        width=-1,
        height=-1,
    )


if __name__ == "__main__":
    typer.run(main)
