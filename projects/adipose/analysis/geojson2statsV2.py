import pickle
import sys
import argparse
import math
import os
import geojson
from shapely.geometry import Polygon, MultiPolygon, shape
import statistics as stats
import numpy as np
from matplotlib import pyplot

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

pp_filter_all = []
area_filter_all = []
area_filter_in_old = []
area_filter_in_new = []

for poly in res:
    area_filter_all.append(poly.area*pixelSize**2)
    pp_filter_all.append((4*math.pi*poly.area ) / ((poly.length)**2))
    if poly.area*pixelSize**2 >= 200 and ((4*math.pi*poly.area ) / ((poly.length)**2)) > 0.75:
        area_filter_in_old.append(poly.area*pixelSize**2)
    if poly.area*pixelSize**2 >= 10**2.5 and ((4*math.pi*poly.area ) / ((poly.length)**2)) > 0.6:
        area_filter_in_new.append(poly.area*pixelSize**2)



bins = np.linspace(200, 20000, 100)

pyplot.figure(figsize=(14, 8))


pyplot.subplot(1, 2, 1)  # First subplot

pyplot.hist(area_filter_all, bins, alpha=0.5, color="black")

pyplot.title("before filter: mean="+str(round(stats.mean(area_filter_all),2))+", med="+str(round(stats.median(area_filter_all),2))+", sd="+str(round(stats.stdev(area_filter_all),2))+", N="+str(len(area_filter_all)))
pyplot.ylabel('counts')
pyplot.xlabel('size (micrometers**2)')

pyplot.subplot(1, 2, 2)  # Second subplot

pyplot.hist(area_filter_in_new, bins, alpha=0.5, color="grey")
pyplot.hist(area_filter_in_old, bins, alpha=0.5, color="grey")
pyplot.ylabel('counts')
pyplot.xlabel('size (micrometers**2)')


pyplot.title("after old filter (black):  mean="+str(round(stats.mean(area_filter_in_old),2))+", med="+str(round(stats.median(area_filter_in_old),2))+", sd="+str(round(stats.stdev(area_filter_in_old),2))+", N="+str(len(area_filter_in_old))+"\n"     
          "after new filter (grey):  mean="+str(round(stats.mean(area_filter_in_new),2))+", med="+str(round(stats.median(area_filter_in_new),2))+", sd="+str(round(stats.stdev(area_filter_in_new),2))+", N="+str(len(area_filter_in_new)))

# Adjust spacing between subplots
pyplot.tight_layout()


pyplot.savefig(filename.replace(".geojson","")+"_size_hist.png")
pyplot.clf()

bins = np.linspace(0, 1, 100)

pyplot.hist(pp_filter_all, bins)
pyplot.title("Histogram of Values")
pyplot.xlabel("size (micrometers**2)")
pyplot.ylabel("counts") 
pyplot.savefig(filename.replace(".geojson","")+"_PP_hist.png")
pyplot.clf()


print("ALL:")
print("mean="+str(round(stats.mean(area_filter_all),2))+", med="+str(round(stats.median(area_filter_all),2))+", sd="+str(round(stats.stdev(area_filter_all),2))+", N="+str(len(area_filter_all)))

print("After old filter (size > 200, PP > 0.75):")
print("mean="+str(round(stats.mean(area_filter_in_old),2))+", med="+str(round(stats.median(area_filter_in_old),2))+", sd="+str(round(stats.stdev(area_filter_in_old),2))+", N="+str(len(area_filter_in_old)))

print("After new filter (size > 316.23, PP > 0.6):")
print("mean="+str(round(stats.mean(area_filter_in_new),2))+", med="+str(round(stats.median(area_filter_in_new),2))+", sd="+str(round(stats.stdev(area_filter_in_new),2))+", N="+str(len(area_filter_in_new)))

