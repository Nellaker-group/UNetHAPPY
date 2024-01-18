import openslide as osl
import typer
import statistics as stats
import math
import shapely
from shapely.geometry import Polygon, MultiPolygon, shape

import projects.adipose.data.geojsoner as gj
import projects.adipose.db.eval_runs_interface as db
from projects.adipose.analysis.get_pixel_size import get_pixel_size, get_which_pixel
from projects.adipose.analysis.db_to_list import db_to_list_within_tile


def main(
    database_id: str = typer.Option(..., help="Which database"),
    eval_run: int = typer.Option(..., help="Which merged Run to convert to .geojson"),
    top_x: int = typer.Option(..., help="x coordinate of top left corner of tile"),
    top_y: int = typer.Option(..., help="y coordinate of top left corner of tile"),
):
    db.init(database_id)

    new_in_list = []
    in_list = []

    # emil str eval_run
    seg_preds = db.get_all_merged_seg_preds(eval_run, eval_run + 1)
    slide_name = db.get_slide_name(eval_run)
    print(slide_name)

    indi_id = slide_name.split("/")[-1]
    used_slide_name = indi_id
    print(indi_id)

    assert len(seg_preds) > 0

    polys = db_to_list_within_tile(seg_preds, top_x, top_y)
    which_pixel = get_which_pixel(indi_id)
    pixel_size = get_pixel_size(which_pixel)

    print("Emil")
    print(len(polys))
    print(len(seg_preds))

    for poly in polys:
        # changed filtering to above 316.23 (10**2.5) in size and PP > 0.6 (07/09/2023)
        if (
            poly.area * pixel_size ** 2 >= 10 ** 2.5
            and ((4 * math.pi * poly.area) / ((poly.length) ** 2)) > 0.6
        ):
            new_in_list.append(poly)
        # removed upper size threshold (07/09/2023)
        if (
            poly.area * pixel_size >= 10 ** 2.5
            and ((4 * math.pi * poly.area) / ((poly.length) ** 2)) > 0.6
        ):
            in_list.append(poly)

    database_id_write = database_id.replace("/", "_")

    gj.writeToGeoJSON(
        polys,
        used_slide_name
        + "_dbID"
        + database_id_write
        + "_evalID"
        + str(eval_run)
        + "_tile_x"
        + str(top_x)
        + "_y"
        + str(top_y)
        + "_ALL.V2.geojson",
    )
    gj.writeToGeoJSON(
        new_in_list,
        f"{used_slide_name}_dbID{database_id_write}_evalID{eval_run}_"
        f"tile_x{top_x}_y{top_y}_newFilterIN.V2.geojson",
    )
    new_in_list_area = [poly.area * pixel_size ** 2 for poly in new_in_list]

    print("After filter (size > 316.23, PP > 0.6):")
    print(
        f"{used_slide_name}, mean={round(stats.mean(new_in_list_area), 2)}, "
        f"med={round(stats.median(new_in_list_area), 2)}, "
        f"sd={round(stats.stdev(new_in_list_area), 2)}, "
        f"N={len(new_in_list_area)}"
    )

    slide = osl.OpenSlide(slide_name)

    cutTile = slide.read_region((top_x, top_y), 0, (1024, 1024))
    cutTile.save(
        used_slide_name
        + "_dbID"
        + database_id_write
        + "_evalID"
        + str(eval_run)
        + "_tile_x"
        + str(top_x)
        + "_y"
        + str(top_y)
        + ".V2.png"
    )


if __name__ == "__main__":
    typer.run(main)
