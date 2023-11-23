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

import openslide as osl



prs = argparse.ArgumentParser()
prs.add_argument('--whichDB', help='Which DB', type=int)
prs.add_argument('--whichEval', help='Which merged Run to convert to .geojson', type=int)
prs.add_argument('--topX', help='X coordinate of top right corner of tile', type=int)
prs.add_argument('--topY', help='Y coordinate of top right corner of tile', type=int)
args = vars(prs.parse_args())

assert args['whichDB'] != ""
assert args['whichEval'] != ""
assert args['topX'] != ""
assert args['topY'] != ""

whichDB = args['whichDB']
whichEval = args['whichEval']
topX = args['topX']
topY = args['topY']


#################################################################

def db_to_list(seg_preds,topX,topY):
    seg_list = []
    poly_list = []
    old_poly_id = seg_preds[0][0]
    tilePoly = Polygon([(topX, topY), (topX+1024, topY), (topX+1024, topY+1024), (topX, topY+1024)])
    for coord in seg_preds:
        if old_poly_id == coord[0]:
            poly_list.append((coord[2],coord[3]))
        else:
            poly=Polygon(poly_list)
            # checking polygon is valid
            if poly.is_valid and poly.within(tilePoly):
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

polys=db_to_list(seg_preds,topX,topY)
    
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
    # changed filtering to above 316.23 (10**2.5) in size and PP > 0.6 (07/09/2023)
    if poly.area*pixels[whichPixel]**2 >= 10**2.5 and ((4*math.pi*poly.area ) / ((poly.length)**2)) > 0.6:
        new_in_list.append(poly)
    # removed upper size threshold (07/09/2023)
    if poly.area*pixels[whichPixel]**2 >= 10**2.5 and ((4*math.pi*poly.area ) / ((poly.length)**2)) > 0.6:
        in_list.append(poly)
        

gj.writeToGeoJSON(polys, used_slide_name+"_DBid"+str(whichDB)+"_evalID"+str(whichEval)+"_tile_x"+str(topX)+"_y"+str(topY)+"_ALL.V2.geojson")
gj.writeToGeoJSON(new_in_list, used_slide_name+"_DBid"+str(whichDB)+"_evalID"+str(whichEval)+"_tile_x"+str(topX)+"_y"+str(topY)+"_newFilterIN.V2.geojson")

new_in_list_area = [poly.area*pixels[whichPixel]**2 for poly in new_in_list]

print("After filter (size > 316.23, PP > 0.6):")
print(used_slide_name+", mean="+str(round(stats.mean(new_in_list_area),2))+", med="+str(round(stats.median(new_in_list_area),2))+", sd="+str(round(stats.stdev(new_in_list_area),2))+", N="+str(len(new_in_list_area)))



slide = osl.OpenSlide(slide_name)

cutTile=slide.read_region((topX,topY),0,(1024,1024))
cutTile.save(used_slide_name+"_DBid"+str(whichDB)+"_evalID"+str(whichEval)+"_tile_x"+str(topX)+"_y"+str(topY)+".V2.png")



