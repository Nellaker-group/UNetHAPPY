from typing import Optional

import pyvips
import typer


def main(
    input_file_name: str = typer.Option(...),
    output_file_name: str = typer.Option(...),
    pixel_size: Optional[float] = None,
    magnification: Optional[float] = None,
):
    """Convert a .tif file into a pyramidal .tiff file adding any missing metadata

    Args:
        input_file_name: location of file to convert
        output_file_name: name to save the new tiff file to
        pixel_size: pixel size float which can be usually found in the metadata
        magnification: max magnification of the slide (40.0 or 20.0 generally)
    """
    original_image = pyvips.Image.new_from_file(input_file_name)
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

    image.tiffsave(
        output_file_name,
        compression="jpeg",
        Q=75,
        tile=True,
        tile_width=1024,
        tile_height=1024,
        bigtiff=True,
        pyramid=True,
    )


if __name__ == "__main__":
    typer.run(main)
