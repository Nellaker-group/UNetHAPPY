import pickle
import sys
import argparse
import math
import os
import geojson
from shapely.geometry import Polygon, MultiPolygon, shape
import statistics as stats

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

area_filter_all = []
area_filter_in = []

for poly in res:
    area_filter_all.append(poly.area*pixelSize**2)
    if poly.area*pixelSize**2 >= 200 and poly.area*pixelSize**2 <= 16000 and ((4*math.pi*poly.area ) / ((poly.length)**2)) > 0.75:
        area_filter_in.append(poly.area*pixelSize**2)



print("ALL:")
print("mean="+str(round(stats.mean(area_filter_all),2))+", med="+str(round(stats.median(area_filter_all),2))+", sd="+str(round(stats.stdev(area_filter_all),2))+", N="+str(len(area_filter_all)))

print("After filter:")
print("mean="+str(round(stats.mean(area_filter_in),2))+", med="+str(round(stats.median(area_filter_in),2))+", sd="+str(round(stats.stdev(area_filter_in),2))+", N="+str(len(area_filter_in)))


