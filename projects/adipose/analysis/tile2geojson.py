import openslide as osl
import typer
import statistics as stats
import math
import shapely
from shapely.geometry import Polygon, MultiPolygon, shape 
 
import projects.adipose.data.merge_polygons as mp
import projects.adipose.data.geojsoner as gj
import projects.adipose.db.eval_runs_interface as db
from projects.adipose.db.eval_runs import EvalRun, TileState, Prediction, UnvalidatedPrediction, MergedPrediction
from projects.adipose.analysis.get_pixel_size import get_pixel_size, get_which_pixel
from projects.adipose.analysis.db_to_list import db_to_list_within_tile

def main(database_id: str = typer.Option(..., help="Which database"), eval_run: int = typer.Option(..., help="Which merged Run to convert to .geojson"), top_x: int = typer.Option(..., help="x coordinate of top right corner of tile"), top_y: int = typer.Option(..., help="y coordinate of top right corner of tile")):
    leipzig = []
    merged_dict_avg = {}
    merged_dict_stdev = {}
    merged_dict_N = {}

    if database_id != None:
        db.init(database_id)
    else:
        db.init()

    new_in_list = []
    in_list = []
    outBig = []
    outSmall = []    
    outBig2 = []
    outSmall2 = []

    # emil str eval_run
    seg_preds = db.get_all_merged_seg_preds(eval_run,eval_run+1)
    slide_name = db.get_slide_name(eval_run)
    used_slide_name = ""
    print(slide_name)

    indi_id = slide_name.split("/")[-1]
    used_slide_name = indi_id
    print(indi_id)

    assert len(seg_preds) > 0

    polys = db_to_list_within_tile(seg_preds,top_x,top_y)
    which_pixel = get_which_pixel(indi_id)
    pixel_size = get_pixel_size(which_pixel)

    print("Emil")
    print(len(polys))
    print(len(seg_preds))

    for poly in polys:
        # changed filtering to above 316.23 (10**2.5) in size and PP > 0.6 (07/09/2023)
        if poly.area*pixel_size**2 >= 10**2.5 and ((4*math.pi*poly.area ) / ((poly.length)**2)) > 0.6:
            new_in_list.append(poly)
        # removed upper size threshold (07/09/2023)
        if poly.area*pixel_size >= 10**2.5 and ((4*math.pi*poly.area ) / ((poly.length)**2)) > 0.6:
            in_list.append(poly)
        
    database_id_write = database_id.replace("/","_")

    gj.writeToGeoJSON(polys, used_slide_name+"_dbID"+database_id_write+"_evalID"+str(eval_run)+"_tile_x"+str(top_x)+"_y"+str(top_y)+"_ALL.V2.geojson")
    gj.writeToGeoJSON(new_in_list, used_slide_name+"_dbID"+database_id_write+"_evalID"+str(eval_run)+"_tile_x"+str(top_x)+"_y"+str(top_y)+"_newFilterIN.V2.geojson")
    new_in_list_area = [poly.area*pixel_size**2 for poly in new_in_list]

    print("After filter (size > 316.23, PP > 0.6):")
    print(used_slide_name+", mean="+str(round(stats.mean(new_in_list_area),2))+", med="+str(round(stats.median(new_in_list_area),2))+", sd="+str(round(stats.stdev(new_in_list_area),2))+", N="+str(len(new_in_list_area)))

    slide = osl.OpenSlide(slide_name)

    cutTile=slide.read_region((top_x,top_y),0,(1024,1024))
    cutTile.save(used_slide_name+"_dbID"+database_id_write+"_evalID"+str(eval_run)+"_tile_x"+str(top_x)+"_y"+str(top_y)+".V2.png")



if __name__ == "__main__":
    typer.run(main)
