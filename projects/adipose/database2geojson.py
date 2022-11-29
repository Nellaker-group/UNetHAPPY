from shapely.geometry import Polygon, MultiPolygon, shape  # (pip install Shapely)

import db.eval_runs_interface as db
import data.geojsoner as gj
import data.merge_polygons as mp
import argparse


prs = argparse.ArgumentParser()
prs.add_argument('--evalRun', help='EvalRun ID run whose polygons to be converted to a .geojson file', type=str)
args = vars(prs.parse_args())
assert args['evalRun'] != ""
evalRun = args['evalRun']

db.init()

seg_preds = db.get_all_validated_seg_preds(evalRun)
poly_list = []

for seg in seg_preds:
    poly=Polygon([(x,y) for x,y in db.stringListTuple2coordinates(seg)])
    poly_list.append(poly)

print("this many merged polygons:")
print(len(poly_list))

slideName = db.get_slide_name(evalRun)
writeName=slideName.split(".")[0]
outputName=writeName+"_runID"+evalRun+".geojson"

print("output name of file is:")
print(outputName)

gj.writeToGeoJSON(poly_list, outputName)
