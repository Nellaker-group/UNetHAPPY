from pathlib import Path
from enum import Enum

import typer
import pandas as pd
import cv2

from nucnet.data.utils import draw_box, draw_centre
from nucnet.utils.enum_args import OrganArg


class ShapeArg(str, Enum):
    box: "box"
    point: "point"


def main(
    image_path: str = typer.Option(...),
    annot_path: str = typer.Option(...),
    shape: ShapeArg = ShapeArg.point,
    organ: OrganArg = OrganArg.placenta,
):
    """Visualises ground truth boxes from annotations for one image

    Args:
        image_path: relative path to image
        annot_path: relative path to annotations
        shape: one of 'box' or 'point'
        organ: one of the supported organs
    """

    all_annotations = pd.read_csv(
        annot_path, names=["image_path", "x1", "y1", "x2", "y2", "class"]
    )

    all_annotations = all_annotations.drop(
        all_annotations[all_annotations["image_path"] != image_path].index
    )
    all_annotations.reset_index(drop=True, inplace=True)

    img = cv2.imread(image_path)

    for i in range(len(all_annotations.index)):
        x1 = all_annotations["x1"][i]
        y1 = all_annotations["y1"][i]
        x2 = all_annotations["x2"][i]
        y2 = all_annotations["y2"][i]
        label_name = all_annotations["class"][i]

        if shape.value == "point":
            draw_centre(img, x1, y1, x2, y2, label_name, cell=False)
        elif shape.value == "box":
            draw_box(img, x1, y1, x2, y2, label_name, cell=False)
        else:
            raise ValueError(f"No such draw shape {shape.value}")

    save_dir = (
        Path(__file__).parent.parent
        / "projects"
        / organ.value
        / "visualisations"
        / f"{image_path.split('/')[-3]}"
        / f"{image_path.split('/')[-2]}"
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    image_name = image_path.split("/")[-1]
    save_path = save_dir / image_name

    print(f"saving to: {save_path}")
    cv2.imwrite(save_path, img)


if __name__ == "__main__":
    typer.run(main)
