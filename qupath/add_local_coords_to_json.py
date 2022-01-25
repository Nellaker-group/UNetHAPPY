from pathlib import Path
import json

import typer


def main(
    project_name: str = typer.Option(...),
    json_filename: str = typer.Option(...),
):
    """Converts QuPath extracted json file of tissue annotation polygons from global
    coordinates to also include local coordinates (coords relative to image tile).

    Args:
        project_name: name of the project directory
        json_filename: path to json file
    """
    projects_dir = Path(__file__).parent.parent / "projects"
    save_dir = projects_dir / project_name / "results" / "tissue_annots"

    with open(save_dir / json_filename) as f:
        data = json.load(f)

    for annotation in data:
        image_name = annotation['image_name']
        name_parts = image_name.split(".")[0].split("_")
        xmin = int(name_parts[1].split("x")[1])
        ymin = int(name_parts[2].split("y")[1])

        coordinates = annotation['coordinates']
        for point in coordinates:
            local_x, local_y = global_coord_to_local(xmin, ymin, point['x'], point['y'])

        print(coordinates)


def global_coord_to_local(xmin, ymin, x, y):
    return x - xmin, y - ymin


if __name__ == "__main__":
    typer.run(main)
