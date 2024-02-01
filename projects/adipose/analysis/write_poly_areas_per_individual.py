import pandas as pd
import statistics as stats
import math
import re
from peewee import fn
import shapely
from shapely.geometry import Polygon, MultiPolygon, shape  # (pip install Shapely)
import typer

import projects.adipose.db.eval_runs_interface as db
import projects.adipose.data.geojsoner as gj
import projects.adipose.data.merge_polygons as mp
from projects.adipose.db.eval_runs import EvalRun, TileState, Prediction, UnvalidatedPrediction, MergedPrediction
from projects.adipose.analysis.db_to_list import db_to_list
from projects.adipose.analysis.get_pixel_size import get_pixel_size, get_which_pixel


def main(i1: int = typer.Option(..., help="Start index (included) of databases"), 
        i2: int = typer.Option(..., help="End index (included) of databases"),
        craig_filter: bool = False,
        old_filter: bool = False, help="Whether to apply filtering like Craig - Whether to apply the previous filtering (size > 200 & PP 0.75 - 07/09/2023)"
    ):
    if craig_filter:
        filter = "Craig"
    elif old_filter:
        filter = "Old"
    else:
        filter = "New"    
    leipzig = []
    merged_dict = {}
    sc_dict = {}
    vc_dict = {}
    file1 = open(f'polygon_areas_sc_fromDB{i1}_toDB{i2}_filtered{filter}.txt', 'w')
    file2 = open(f'polygon_areas_vc_fromDB{i1}_toDB{i2}_filtered{filter}.txt', 'w')
    endoxID = pd.read_csv("/gpfs3/well/lindgren/craig/isbi-2012/NDOG_histology_IDs_and_quality.csv")
    endoxID_sc = endoxID[ endoxID["depot"] == "subcutaneous"]
    for j in range(i1,i2+1):
        db.init(f"task_dbs/{j}.db")
        max_run_id = EvalRun.select(fn.MAX(EvalRun.id)).scalar()
        for i in range(1,max_run_id,2):
            sc_list = []
            vc_list = []
            below750_vc_list = []
            below750_sc_list = []
            seg_preds = db.get_all_merged_seg_preds(i,i+1)
            slide_name = db.get_slide_name(i)
            used_slide_name = ""
            print(slide_name)
            indi_id = slide_name.split("/")[-1]
            used_slide_name = indi_id
            print(indi_id)        
            sub = False
            if re.search(r"\dsc\d", indi_id) or "Subcutaneous" in indi_id or any(indi_id == endoxID_sc["filename"]):
                indi_id2 = indi_id.split("sc")[0]
                sub = True
            elif re.search(r"\dvc\d", indi_id) or "Visceral" in indi_id:
                indi_id2 = indi_id.split("vc")[0]
            else:
                indi_id2 = indi_id
            if len(seg_preds) == 0:
                continue
            polys=db_to_list(seg_preds)
            which_pixel = get_which_pixel(indi_id)
            pixel_size = get_pixel_size(which_pixel)
            for poly in polys:
                if craig_filter:
                    good = poly.area*pixel_size**2 >= 200 and poly.area*pixel_size**2 <= 16000
                elif old_filter:
                    good = poly.area*pixel_size**2 >= 200 and ((4*math.pi*poly.area ) / ((poly.length)**2)) > 0.75
                else:
                    good = poly.area*pixel_size**2 >= 10**2.5 and ((4*math.pi*poly.area ) / ((poly.length)**2)) > 0.6
                if good:
                    if sub:
                        sc_list.append(poly.area*pixel_size**2)
                    else:
                        vc_list.append(poly.area*pixel_size**2)
                    if sub and poly.area*pixel_size**2 < 750:
                        below750_sc_list.append(poly.area*pixel_size**2)
                    elif poly.area*pixel_size**2 < 750:
                        below750_vc_list.append(poly.area*pixel_size**2)
            if len(sc_list) > 0:
                # Convert the numbers in the list to strings and join them with commas
                fracBelow750 = len(below750_sc_list) / len(sc_list)
                line = ",".join(str(num) for num in sc_list)
                # Write the line to the file
                file1.write(f"{which_pixel},{slide_name},{len(sc_list)},{fracBelow750},{line}\n")
            else:
                fracBelow750 = 0
                file1.write(f"{which_pixel},{slide_name},{len(sc_list)},{fracBelow750},NA\n")
            if len(vc_list) > 0:
                # Convert the numbers in the list to strings and join them with commas
                fracBelow750 = len(below750_vc_list) / len(vc_list)
                line = ",".join(str(num) for num in vc_list)
                # Write the line to the file
                file2.write(f"{which_pixel},{slide_name},{len(vc_list)},{fracBelow750},{line}\n")
            else:
                fracBelow750 = 0
                file2.write(f"{which_pixel},{slide_name},{len(vc_list)},{fracBelow750},NA\n")            
    file1.close()
    file2.close()

if __name__ == "__main__":
    typer.run(main)


