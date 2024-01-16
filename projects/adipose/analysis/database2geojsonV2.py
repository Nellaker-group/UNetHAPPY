from shapely.geometry import Polygon, MultiPolygon, shape  # (pip install Shapely)
import typer

import projects.adipose.db.eval_runs_interface as db
import projects.adipose.data.geojsoner as gj
import projects.adipose.data.merge_polygons as mp
from projects.adipose.analysis.db_to_list import db_to_list


def main(database_id: str = typer.Option(..., help='EvalRun ID run whose polygons to be converted to a .geojson file'), eval_run: int = typer.Option(..., help='EvalRun ID run whose polygons to be converted to a .geojson file')):
    if database_id != None:
        db.init(database_id)
    else:
        db.init()

    seg_preds = db.get_all_validated_seg_preds(eval_run)
    polys = db_to_list(seg_preds)

    print("this many merged polygons:")
    print(len(polys))

    slide_name = db.get_slide_name(eval_run)
    write_name = slide_name.split(".")[0]
    output_name = write_name+"_runID"+eval_run+".geojson"

    print("output name of file is:")
    print(output_name)

    gj.writeToGeoJSON(polys, output_name)

if __name__ == "__main__":
    typer.run(main)
