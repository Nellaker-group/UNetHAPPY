import typer
import numpy as np

from happy.graph.create_graph import get_groundtruth_patch
from happy.graph.utils.visualise_points import visualize_points
from happy.utils.utils import get_project_dir
from happy.organs import get_organ


def main(
    project_name: str = "placenta",
    organ_name: str = "placenta",
    x_min: int = 0,
    y_min: int = 0,
    width: int = -1,
    height: int = -1,
    tissue_label_tsv: str = "139_tissue_points.tsv",
    remove_unlabelled: bool = False,
    slide_name: str = "slide_139-2019-09-09 11.15.35.ndpi",
):
    project_dir = get_project_dir(project_name)
    organ = get_organ(organ_name)

    xs, ys, tissue_class = get_groundtruth_patch(
        organ, project_dir, x_min, y_min, width, height, tissue_label_tsv
    )

    if remove_unlabelled:
        labelled_inds = tissue_class.nonzero()
        tissue_class = tissue_class[labelled_inds]
        xs = xs[labelled_inds]
        ys = ys[labelled_inds]

    unique, counts = np.unique(tissue_class, return_counts=True)
    print(dict(zip(unique, counts)))

    save_dir = (
        project_dir / "visualisations" / "graphs" / "lab_1" / slide_name / "groundtruth"
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    plot_name = f"x{x_min}_y{y_min}_w{width}_h{height}"
    save_path = save_dir / plot_name

    colours_dict = {tissue.id: tissue.colour for tissue in organ.tissues}
    colours = [colours_dict[label] for label in tissue_class]
    visualize_points(
        organ,
        save_path,
        np.stack((xs, ys), axis=1),
        colours=colours,
        width=width,
        height=height,
    )


if __name__ == "__main__":
    typer.run(main)
