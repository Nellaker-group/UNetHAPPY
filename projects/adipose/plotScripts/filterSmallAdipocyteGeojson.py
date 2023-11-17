import pickle
import sys
import argparse
import math
import os
import geojson
from shapely.geometry import Polygon, MultiPolygon, shape
from matplotlib import pyplot
import numpy as np
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

print("This many polygons before filtering:")
print(len(res))

pp = []
area = []
area_filter_normal = []
res_filter_PPabove = []
res_filter_PPbelow = []

for poly in res:
    pp.append(((4*math.pi*poly.area ) / ((poly.length)**2)))
    area.append(poly.area*pixelSize**2)
    if poly.area*pixelSize**2 >= 200 and poly.area*pixelSize**2 <= 16000 and ((4*math.pi*poly.area ) / ((poly.length)**2)) > 0.75:
        area_filter_normal.append(poly.area*pixelSize**2)
    if poly.area*pixelSize**2 >= 200 and poly.area*pixelSize**2 <= 500 and ((4*math.pi*poly.area ) / ((poly.length)**2)) > 0.75:
        res_filter_PPabove.append(poly)
    elif poly.area*pixelSize**2 >= 200 and poly.area*pixelSize**2 <= 500 and ((4*math.pi*poly.area ) / ((poly.length)**2)) < 0.75:
        res_filter_PPbelow.append(poly)

print("This many polygons after filtering:")
print(len(area_filter_normal))

counter = 0
for ele in area_filter_normal:
    if ele < 750:
        counter += 1

print("This many filtered below 400 (area):")
print(counter / len(area_filter_normal))

writeToGeoJSON(res_filter_PPbelow, filename.replace(".geojson","_SMALL.belowPP.geojson"))
writeToGeoJSON(res_filter_PPabove, filename.replace(".geojson","_SMALL.abovePP.geojson"))

bins = np.linspace(200, 16000, 100)
pyplot.hist(area_filter_normal, bins, alpha=0.5, color="black")
pyplot.title("size: mean="+str(round(stats.mean(area_filter_normal),2))+", med="+str(round(stats.median(area_filter_normal),2))+", sd="+str(round(stats.stdev(area_filter_normal),2))+", N="+str(len(area_filter_normal)))
pyplot.xlabel('size (micrometers**2)')
pyplot.ylabel('counts')
pyplot.savefig(filename.replace(".geojson","_hist.afterFilter.png"))
pyplot.clf()


bins = np.linspace(0, 1, 100)
pyplot.hist(pp, bins, alpha=0.5, color="black")
pyplot.title("All adipocytes from slide")
pyplot.xlabel('PP')
pyplot.ylabel('counts')
pyplot.savefig(filename.replace(".geojson","_hist.PP.beforeFilter.png"))
pyplot.clf()


bins = np.linspace(0, 25000, 100)
pyplot.hist(area, bins, alpha=0.5, color="black")
pyplot.title("All adipocytes from slide")
pyplot.xlabel('size (micrometers**2)')
pyplot.ylabel('counts')
pyplot.savefig(filename.replace(".geojson","_hist.beforeFilter.png"))
pyplot.clf()


bins = np.linspace(200, 2000, 20)
pyplot.hist(area_filter_normal, bins, alpha=0.5, color="black")
pyplot.title("size: mean="+str(round(stats.mean(area_filter_normal),2))+", med="+str(round(stats.median(area_filter_normal),2))+", sd="+str(round(stats.stdev(area_filter_normal),2))+", N="+str(len(area_filter_normal)))
pyplot.xlabel('size (micrometers**2)')
pyplot.ylabel('counts')
pyplot.savefig(filename.replace(".geojson","_hist.afterFilter.zoomed.png"))
pyplot.clf()
