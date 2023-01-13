from typing import Optional
from pathlib import Path

import pyvips
import typer


def main(
    input_file_name: Path = typer.Option(
        ..., exists=True, file_okay=True, dir_okay=False
    ),
    output_file_name: Path = typer.Option(...),
    pixel_size: Optional[float] = None,
    magnification: Optional[float] = None,
):
    """Convert a .tif file into a pyramidal .tiff file and generate a thumbnail.

    Args:
        input_file_name: location of file to convert
        output_file_name: name to save the new tiff file to
        pixel_size: pixel size float which can be usually found in the metadata
        magnification: max magnification of the slide (40.0 or 20.0 generally)
    """
    original_image = pyvips.Image.new_from_file(str(input_file_name))
    image = original_image.copy()

    image_fields = image.get_fields()

    if "openslide.mpp-x" not in image_fields:
        assert pixel_size
        image.set_type(pyvips.GValue.gdouble_type, "openslide.mpp-x", pixel_size)
    if "openslide.mpp-y" not in image_fields:
        assert pixel_size
        image.set_type(pyvips.GValue.gdouble_type, "openslide.mpp-y", pixel_size)
    if "openslide.objective-power" not in image_fields:
        assert magnification
        image.set_type(
            pyvips.GValue.gdouble_type, "openslide.objective-power", magnification
        )

    # Save the image as a pyramidal bigtiff
    split_path = list(output_file_name.parts)
    file_name = split_path[-1]
    split_path[-1] = f"pyr_{file_name}"
    split_path.insert(-1, "pyr")
    pyr_output_path = str(Path(*split_path))

    image.tiffsave(
        pyr_output_path,
        compression="jpeg",
        Q=75,
        tile=True,
        tile_width=1024,
        tile_height=1024,
        bigtiff=True,
        pyramid=True,
    )

    # Read the bigtiff and generate thumbnail (faster than using the original tif)
    thumbnail = pyvips.Image.thumbnail(pyr_output_path, 4000)
    split_path[-1] = f"tn_{file_name.split('.tiff')[0]}.jpg"
    split_path[-2] = "thumb"
    thumb_output_path = str(Path(*split_path))
    thumbnail.write_to_file(thumb_output_path, Q=90, optimize_coding=True, strip=True)


if __name__ == "__main__":
    typer.run(main)
