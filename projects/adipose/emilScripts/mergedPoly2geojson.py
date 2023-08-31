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
prs.add_argument('--whichDB', help='Which DB', type=int)
prs.add_argument('--whichEval', help='Which merged Run to convert to .geojson', type=int)
args = vars(prs.parse_args())

assert args['whichDB'] != ""
assert args['whichEval'] != ""

whichDB = args['whichDB']
whichEval = args['whichEval']

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


db.init(str(whichDB).replace(str(whichDB),"task_dbs/"+str(whichDB)+".db"))




new_in_list = []
in_list = []
outBig = []
outSmall = []    
outBig2 = []
outSmall2 = []


seg_preds = db.get_all_merged_seg_preds(whichEval,whichEval+1)
slide_name = db.get_slide_name(whichEval)
used_slide_name = ""
print(slide_name)

indi_id = slide_name.split("/")[-1]
used_slide_name = indi_id

print(indi_id)

assert len(seg_preds) > 0

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
        new_in_list.append(poly)
    if poly.area*pixels[whichPixel]**2 >= 200 and poly.area*pixels[whichPixel]**2 <= 16000 and ((4*math.pi*poly.area ) / ((poly.length)**2)) > 0.75:
        in_list.append(poly)
    elif  poly.area*pixels[whichPixel]**2 > 16000 and ((4*math.pi*poly.area ) / ((poly.length)**2)) > 0.75:
        outBig.append(poly)
    elif  poly.area*pixels[whichPixel]**2 < 200 and ((4*math.pi*poly.area ) / ((poly.length)**2)) > 0.75:
        outSmall.append(poly)
    elif  poly.area*pixels[whichPixel]**2 > 16000 and ((4*math.pi*poly.area ) / ((poly.length)**2)) < 0.75:
        outBig2.append(poly)
    elif  poly.area*pixels[whichPixel]**2 < 200 and ((4*math.pi*poly.area ) / ((poly.length)**2)) < 0.75:
        outSmall2.append(poly)
        

gj.writeToGeoJSON(polys, used_slide_name+"_DBid"+str(whichDB)+"_evalID"+str(whichEval)+"_ALL.geojson")
gj.writeToGeoJSON(new_in_list, used_slide_name+"_DBid"+str(whichDB)+"_evalID"+str(whichEval)+"_newFilterIN.geojson")
gj.writeToGeoJSON(in_list, used_slide_name+"_DBid"+str(whichDB)+"_evalID"+str(whichEval)+"_IN.geojson")
gj.writeToGeoJSON(outBig, used_slide_name+"_DBid"+str(whichDB)+"_evalID"+str(whichEval)+"_OUTbigPPok.geojson")
gj.writeToGeoJSON(outSmall, used_slide_name+"_DBid"+str(whichDB)+"_evalID"+str(whichEval)+"_OUTsmallPPok.geojson")
gj.writeToGeoJSON(outBig2, used_slide_name+"_DBid"+str(whichDB)+"_evalID"+str(whichEval)+"_OUTbigPPbad.geojson")
gj.writeToGeoJSON(outSmall2, used_slide_name+"_DBid"+str(whichDB)+"_evalID"+str(whichEval)+"_OUTsmallPPbad.geojson")



