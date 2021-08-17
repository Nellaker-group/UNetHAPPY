import typer
import matplotlib.pyplot as plt

from projects.placenta.graphs.graphs.create_graph import get_groundtruth_patch
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

    fig = plt.figure(figsize=(8, 8), dpi=150)
    plt.scatter(xs, ys, marker=".", s=1, zorder=1000, c=tissue_class, cmap="Spectral")
    plt.gca().invert_yaxis()
    plt.axis("off")
    fig.tight_layout()
    plt.savefig(save_path)


if __name__ == "__main__":
    typer.run(main)
