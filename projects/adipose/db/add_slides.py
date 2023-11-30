# import sys
import os
from pathlib import Path

# sys.path.append(os.getcwd())

import typer

from db.slides import Slide, Lab
import db.eval_runs_interface as db


def main(
    slides_dir: Path = typer.Option(
        ..., exists=True, file_okay=False, dir_okay=True, resolve_path=True
    ),
    lab_country: str = typer.Option(...),
    primary_contact: str = typer.Option(...),
    slide_file_format: str = typer.Option(...),
    pixel_size: float = typer.Option(...),
    has_notes: bool = False,
    has_clinical_data: bool = False,
    db_name: str = "main.db",
    avoid_keyword: str = "",

):
    """Add a whole lab of slides to the database

    Args:
        slides_dir: absolute path to the dir containing the slides - the script will crawl through every file and subfolder using os.walk - only adding files with the corret slide_file_format
        lab_country: country where the lab is
        primary_contact: first name of collaborator
        slide_file_format: file format of slides, e.g. '.svs'
        avoid_keyword: it will avoid files with that keyword in their filename
        pixel_size: pixel size of all slides. Can be found with QuPath on one slide
        has_notes: if the slides came with associated pathologist's notes
        has_clinical_data: if the slides came with associated clinical data/history
        db_name: name of the database or .db file being written to
    """
    db.init(db_name)

    lab = Lab.create(
        country=lab_country,
        primary_contact=primary_contact,
        slides_dir=slides_dir,
        has_pathologists_notes=has_notes,
        has_clinical_data=has_clinical_data,
    )

    for path, dirs, files in os.walk(slides_dir):
        for filename in files:
            if avoid_keyword != "":
                if filename.endswith(slide_file_format) and not avoid_keyword in filename:
                    fullfilename = path + "/" + filename
                    Slide.create(slide_name=fullfilename, pixel_size=pixel_size, lab=lab)
            else:
                if filename.endswith(slide_file_format):
                    fullfilename = path + "/" + filename
                    Slide.create(slide_name=fullfilename, pixel_size=pixel_size, lab=lab)


if __name__ == "__main__":
    typer.run(main)
