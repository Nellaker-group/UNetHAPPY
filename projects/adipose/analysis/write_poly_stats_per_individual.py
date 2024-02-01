import pandas as pd
import statistics as stats
import math
import shapely
from shapely.geometry import Polygon, MultiPolygon, shape  # (pip install Shapely)
from peewee import fn
import typer

import projects.adipose.db.eval_runs_interface as db
import projects.adipose.data.geojsoner as gj
import projects.adipose.data.merge_polygons as mp
from projects.adipose.db.eval_runs import EvalRun, TileState, Prediction, UnvalidatedPrediction, MergedPrediction
from projects.adipose.analysis.db_to_list import db_to_list
from projects.adipose.analysis.get_pixel_size import get_pixel_size, get_which_pixel

def main(  
    i1: int = typer.Option(..., help="Start index (included) of databases"),
    i2: int = typer.Option(..., help="End index (included) of databases"),
    craig_filter: bool = False, help="Whether to apply filtering like Craig"
):
    merged_dict_avg = {}
    merged_dict_stdev = {}
    merged_dict_N = {}
    merged_dict_below750 = {}          
    if craig_filter:
        filter = "Craig"
    else:
        filter = "New"    
    for j in range(i1,i2+1):
        db.init(f"task_dbs/{j}.db")
        max_run_id = EvalRun.select(fn.MAX(EvalRun.id)).scalar()
        for i in range(1,max_run_id,2):
            poly_list = []
            below750_list = []
            seg_preds = db.get_all_merged_seg_preds(i,i+1)
            slide_name = db.get_slide_name(i)
            used_slide_name = ""
            print(slide_name)
            indi_id = slide_name.split("/")[-1]
            used_slide_name = indi_id
            print(indi_id)
            if len(seg_preds) == 0:
                continue
            polys=db_to_list(seg_preds)
            which_pixel = get_which_pixel(indi_id)
            pixel_size = get_pixel_size(which_pixel)
            for poly in polys:
                if craig_filter:
                    good = poly.area*pixel_size**2 >= 200 and poly.area*pixel_size**2 <= 16000
                else:
                    good = poly.area*pixel_size**2 >= 10**2.5 and ((4*math.pi*poly.area ) / ((poly.length)**2)) > 0.60
                if good:
                    poly_list.append(poly.area*pixel_size**2)
                    if poly.area*pixel_size**2 < 750:
                        below750_list.append(poly.area*pixel_size**2)
            if len(poly_list) > 1:
                merged_dict_avg[indi_id] = stats.mean(poly_list)
                merged_dict_stdev[indi_id] = stats.stdev(poly_list)
                merged_dict_N[indi_id] = len(poly_list)
                merged_dict_below750[indi_id] = len(below750_list) / len(poly_list)
            else:
                merged_dict_avg[indi_id] = -9
                merged_dict_stdev[indi_id] = -9
                merged_dict_N[indi_id] = len(poly_list)
                merged_dict_below750[indi_id] = len(below750_list)

        df = pd.DataFrame.from_dict(merged_dict_avg, orient="index", columns=["avg"])
        df2 = pd.DataFrame.from_dict(merged_dict_stdev, orient="index", columns=["stdev"])
        df3 = pd.DataFrame.from_dict(merged_dict_N, orient="index", columns=["Nadipocytes"])
        df4 = pd.DataFrame.from_dict(merged_dict_below750, orient="index", columns=["fracBelow750"])
        dfTotal=pd.concat([df['avg'],df2['stdev'],df3['Nadipocytes'],df4['fracBelow750']],axis=1)
        dfTotal.to_csv(f"avg_stdev_Nadipocytes_fracBelow750_fromDB{i1}_toDB{i2}_filtered{filter}.csv",index_label="slide_name")


if __name__ == "__main__":
    typer.run(main)


