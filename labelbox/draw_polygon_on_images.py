from pathlib import Path
import json

import typer
from tqdm import tqdm
from PIL import ImageDraw
from PIL import Image

from convert_global_to_local_coords import convert_global_to_local


def main(
    images_dir: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True),
    json_filename: str = typer.Option(...),
    subregion: bool = False
):
    """Generates duplicate images with a polygon drawn on them

    Args:
        images_dir: Path to dir containing original images
        json_filename: Name of annotations json file
    """
    save_dir = images_dir / "polygon_images"
    save_dir.mkdir(exist_ok=True)

    with open(images_dir / json_filename) as f:
        data = json.load(f)

    for annotation in tqdm(data):
        image_name = annotation["image_name"]
        image_path = images_dir / image_name
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image, 'RGBA')

        local_polygon_coords = convert_global_to_local(annotation, subregion)

        converted_coords = []
        for coord in local_polygon_coords:
            converted_coords.append((coord["x"], coord["y"]))

        draw.polygon(converted_coords, fill=(0, 225, 0, 50), outline=(0, 225, 0))
        image.save(save_dir / image_name)


if __name__ == "__main__":
    typer.run(main)
