import pickle
import sys
import argparse
import math
import os
import geojson
from shapely.geometry import Polygon, MultiPolygon, shape

sys.path.append(os.getcwd())
import db.eval_runs_interface as db
from data.geojsoner import  writeToGeoJSON, geojson2polygon, readGeoJSON2list



prs = argparse.ArgumentParser()
prs.add_argument('--filename', help='Filename of geojson file to filter', type=str)
prs.add_argument('--pixel_size', help='Pixel size fo calculating the area', type=float)
args = vars(prs.parse_args())

assert args['filename'] != ""
assert args['pixel_size'] != ""

filename = args['filename']
pixelSize = args['pixel_size']

############################################

res = readGeoJSON2list(filename)

res_filter_all = []
res_filter_in = []
res_filter_out = []

for poly in res:
    res_filter_all.append(poly)
    if poly.area*pixelSize**2 >= 200 and poly.area*pixelSize**2 <= 16000 and ((4*math.pi*poly.area ) / ((poly.length)**2)) > 0.75:
        res_filter_in.append(poly)
    else:
        res_filter_out.append(poly)

writeToGeoJSON(res_filter_all, filename.replace(".geojson","_filteredALL.geojson"))
writeToGeoJSON(res_filter_in, filename.replace(".geojson","_filteredIN.geojson"))
writeToGeoJSON(res_filter_out, filename.replace(".geojson","_filteredOUT.geojson"))
