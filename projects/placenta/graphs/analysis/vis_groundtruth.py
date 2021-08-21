import typer
import numpy as np

from projects.placenta.graphs.graphs.create_graph import get_groundtruth_patch
from projects.placenta.graphs.analysis.vis_graph_patch import visualize_points
from happy.utils.utils import get_project_dir
from happy.organs.organs import get_organ


def main(
    project_name: str = "placenta",
    organ_name: str = "placenta",
    x_min: int = 0,
    y_min: int = 0,
    width: int = -1,
    height: int = -1,
):
    project_dir = get_project_dir(project_name)
    organ = get_organ(organ_name)

    xs, ys, tissue_class = get_groundtruth_patch(
        organ, project_dir, x_min, y_min, width, height
    )

    save_dir = (
        project_dir
        / "visualisations"
        / "graphs"
        / "lab_1"
        / "slide_139-2019-09-09 11.15.35.ndpi"
        / "groundtruth"
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    plot_name = f"x{x_min}_y{y_min}_w{width}_h{height}"
    save_path = save_dir / plot_name

    visualize_points(
        organ,
        save_path,
        np.stack((xs, ys), axis=1),
        colours=tissue_class,
    )


if __name__ == "__main__":
    typer.run(main)
