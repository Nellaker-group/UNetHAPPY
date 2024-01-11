from shapely.geometry import Polygon
import pickle
import db.eval_runs_interface as db
from data.geojsoner import  writeToGeoJSON
import argparse
import data.merge_polygons as mp

prs = argparse.ArgumentParser()
prs.add_argument('--pickleFile', help='pickle file to convert into a geojson file - should end with .obj', type=str)
prs.add_argument('--overlap', help='pickle file to convert into a geojson file - should end with .obj', type=bool, default=True)
args = vars(prs.parse_args())
assert args['pickleFile'] != "" 

pickleFile = args['pickleFile']
overlap = args['overlap']


file = open(pickleFile,'rb')
poly_preds = pickle.load(file)
file.close()
poly_list = []


for poly in poly_preds:
    poly_list.append(poly)


geojsonFile = pickleFile.replace(".obj",".geojson")


writeToGeoJSON(poly_list, geojsonFile)


merged_polys_list = []
if overlap:            
    merged_polys_list = mp.merge_polysV3(poly_list)

geojsonFile2 = pickleFile.replace(".obj","_merged.geojson")

writeToGeoJSON(merged_polys_list, geojsonFile2)
