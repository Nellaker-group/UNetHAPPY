import os
import openslide as osl
import typer


def main(slide_name: str = typer.Option(...), thumbnail_size: int = 4096):
    slide = osl.OpenSlide(slide_name)
    thumb = slide.get_thumbnail((thumbnail_size, thumbnail_size))
    thumb.save(
        os.path.basename(
            slide_name.replace(".svs", "_thumbnail.png").replace(
                ".scn", "_thumbnail.png"
            )
        )
    )


if __name__ == "__main__":
    typer.run(main)
