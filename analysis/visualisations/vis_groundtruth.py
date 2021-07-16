from enum import Enum
from pathlib import Path

import cv2
import pandas as pd
import typer

from happy.data.utils import draw_box, draw_centre
from happy.cells.cells import get_organ


class ShapeArg(str, Enum):
    box = "box"
    point = "point"


def main(
    project_name: str = typer.Option(...),
    organ_name: str = typer.Option(...),
    image_path: str = typer.Option(...),
    annot_path: str = typer.Option(...),
    shape: ShapeArg = ShapeArg.point,
):
    """Visualises ground truth boxes or points from annotations for one image
    Args:
        project_name: name of the project dir to save visualisations to
        organ_name: name of organ
        image_path: relative path to image
        annot_path: relative path to annotations
        shape: one of 'box' or 'point' for visualising the prediction
    """
    organ = get_organ(organ_name)

    project_dir = Path(__file__).parent.parent.parent / "projects" / project_name
    annotation_path = project_dir / annot_path

    all_annotations = pd.read_csv(
        annotation_path, names=["image_path", "x1", "y1", "x2", "y2", "class"]
    )
    image_annotations = all_annotations[
        all_annotations["image_path"] == image_path
    ].reset_index(drop=True)

    img = cv2.imread(str(project_dir / image_path))

    for i in range(len(image_annotations.index)):
        x1 = image_annotations["x1"][i]
        y1 = image_annotations["y1"][i]
        x2 = image_annotations["x2"][i]
        y2 = image_annotations["y2"][i]
        label_name = image_annotations["class"][i]

        if shape.value == "point":
            draw_centre(img, x1, y1, x2, y2, label_name, organ, cell=False)
        elif shape.value == "box":
            draw_box(img, x1, y1, x2, y2, label_name, organ, cell=False)
        else:
            raise ValueError(f"No such draw shape {shape.value}")

    save_dir = (
        project_dir
        / "visualisations"
        / f"{image_path.split('/')[-3]}"
        / f"{image_path.split('/')[-2]}_gt"
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    image_name = image_path.split("/")[-1]
    save_path = save_dir / image_name

    print(f"saving to: {save_path}")
    cv2.imwrite(str(save_path), img)


if __name__ == "__main__":
    typer.run(main)
