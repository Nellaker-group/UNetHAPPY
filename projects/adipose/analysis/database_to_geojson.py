from shapely.geometry import Polygon, MultiPolygon, shape  # (pip install Shapely)
import typer
import math

import projects.adipose.db.eval_runs_interface as db
import projects.adipose.data.geojsoner as gj
import projects.adipose.data.merge_polygons as mp
from projects.adipose.analysis.db_to_list import db_to_list
from projects.adipose.analysis.get_pixel_size import get_pixel_size, get_which_pixel


def main(database_id: str = typer.Option(..., help='eval_run ID run whose polygons to be converted to a .geojson file'),
         eval_run: int = typer.Option(..., help='EvalRun ID run whose polygons to be converted to a .geojson file'),
         merge_runs: bool = False, help="To you want to merge polygons from the subsequent eval_run ID (e.g. 3 and 4) - has to be uneven number!",
         filter_polys: bool = True):
    if database_id != None:
        db.init(database_id)
    else:
        db.init()
    if merge_runs:
        # checking that eval run is an odd number - if merging runs - as it will merge eval_run and eval_run+1
        assert eval_run % 2 != 0 and eval_run > 0
        seg_preds = db.get_all_merged_seg_preds(eval_run,eval_run+1)
    else:
        seg_preds = db.get_all_validated_seg_preds(eval_run)
    polys = db_to_list(seg_preds)
    print("this many merged polygons:")
    print(len(polys))
    slide_name = db.get_slide_name(eval_run)
    write_name = slide_name.split(".")[0].split("/")[-1]
    if merge_runs:
        output_name = write_name+"_evalID"+str(eval_run)+"_"+str(eval_run+1)+".geojson"
    else:
        output_name = write_name+"_evalID"+str(eval_run)+".geojson"
    print("output name of file is:")
    print(output_name)
    if filter_polys:
        new_in_list = []
        indi_id = slide_name.split("/")[-1]
        which_pixel = get_which_pixel(indi_id)
        pixel_size = get_pixel_size(which_pixel)
        for poly in polys:
            if poly.area*pixel_size**2 >= 10**2.5 and ((4*math.pi*poly.area ) / ((poly.length)**2)) > 0.6:
                new_in_list.append(poly)
        print("this many polygons - after filtering:")
        print(len(polys))
    gj.writeToGeoJSON(polys, output_name)

if __name__ == "__main__":
    typer.run(main)
