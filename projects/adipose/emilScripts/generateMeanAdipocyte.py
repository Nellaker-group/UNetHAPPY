import os
print(os.environ["CONDA_PREFIX"])
print(os.environ['CONDA_DEFAULT_ENV'])

import sys
import os
# to be able to read in the libaries properly
sys.path.append(os.getcwd())

import pandas as pd
import statistics as stats
import math
import db.eval_runs_interface as db
import data.geojsoner as gj
import data.merge_polygons as mp
import argparse
from peewee import fn
from db.eval_runs import EvalRun, TileState, Prediction, UnvalidatedPrediction, MergedPrediction


import shapely
from shapely.geometry import Polygon, MultiPolygon, shape  # (pip install Shapely)


prs = argparse.ArgumentParser()
prs.add_argument('--i1', help='Start index (included)', type=int)
prs.add_argument('--i2', help='End index (included)', type=int)
args = vars(prs.parse_args())

assert args['i1'] != ""
assert args['i2'] != ""

i1 = args['i1']
i2 = args['i2']


#################################################################

def db_to_list(seg_preds):
    seg_list = []
    poly_list = []
    old_poly_id = seg_preds[0][0]
    for coord in seg_preds:
        if old_poly_id == coord[0]:
            poly_list.append((coord[2],coord[3]))
        else:
            poly=Polygon(poly_list)
            # checking polygon is valid
            if poly.is_valid:
                seg_list.append(poly)
            poly_list = []
            # for the first point when poly_id changes
            poly_list.append((coord[2],coord[3]))
            old_poly_id = coord[0]
    return(seg_list)

#################################################################

pixels = {}

pixels["leipzig"] = 0.5034
pixels["munich"] = 0.5034
pixels["hohenheim"] = 0.5034
pixels["gtex"] = 0.4942
pixels["endox"] = 0.2500

leipzig = []

merged_dict_avg = {}
merged_dict_stdev = {}
merged_dict_N = {}

for j in range(i1,i2+1):

    db.init(str(j).replace(str(j),"task_dbs/"+str(j)+".db"))

    max_run_id = EvalRun.select(fn.MAX(EvalRun.id)).scalar()

    for i in range(1,max_run_id,2):

        tmp_list = []

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
                
        whichPixel = ""
        if indi_id.startswith("a"):
            whichPixel = "leipzig"
        elif indi_id.startswith("m"):
            whichPixel = "munich"
        elif indi_id.startswith("h"):
            whichPixel = "hohenheim"
        elif indi_id.startswith("GTEX"):
            whichPixel = "gtex"
        elif indi_id.startswith("Image"):
            whichPixel = "endox"

        for poly in polys:
            if poly.area*pixels[whichPixel]**2 >= 200 and ((4*math.pi*poly.area ) / ((poly.length)**2)) > 0.75:
                tmp_list.append(poly.area*pixels[whichPixel]**2)
        
        if len(tmp_list) > 1:
            merged_dict_avg[indi_id] = stats.mean(tmp_list)
            merged_dict_stdev[indi_id] = stats.stdev(tmp_list)
            merged_dict_N[indi_id] = len(tmp_list)
        else:
            merged_dict_avg[indi_id] = -9
            merged_dict_stdev[indi_id] = -9
            merged_dict_N[indi_id] = len(tmp_list)


    df = pd.DataFrame.from_dict(merged_dict_avg, orient="index", columns=["avg"])
    df2 = pd.DataFrame.from_dict(merged_dict_stdev, orient="index", columns=["stdev"])
    df3 = pd.DataFrame.from_dict(merged_dict_N, orient="index", columns=["Nadipocytes"])
    df.to_csv("multiRun_avg_from"+str(i1)+"_to"+str(i2)+"V2.csv")
    df2.to_csv("multiRun_stdev_from"+str(i1)+"_to"+str(i2)+"V2.csv")
    df3.to_csv("multiRun_Nadipocytes_from"+str(i1)+"_to"+str(i2)+"V2.csv")


