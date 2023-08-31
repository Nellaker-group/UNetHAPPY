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
area_filter_in = []

for poly in res:
    area_filter_all.append(poly.area*pixelSize**2)
    pp_filter_all.append((4*math.pi*poly.area ) / ((poly.length)**2))
    if poly.area*pixelSize**2 >= 200 and ((4*math.pi*poly.area ) / ((poly.length)**2)) > 0.75:
        area_filter_in.append(poly.area*pixelSize**2)



bins = np.linspace(200, 20000, 100)

fig, axs = pyplot.subplots(2)
fig.suptitle("merged-size: mean="+str(round(stats.mean(area_filter_all),2))+", med="+str(round(stats.median(area_filter_all),2))+", sd="+str(round(stats.stdev(area_filter_all),2))+", N="+str(len(area_filter_all))+"\n"
             "size: mean="+str(round(stats.mean(area_filter_in),2))+", med="+str(round(stats.median(area_filter_in),2))+", sd="+str(round(stats.stdev(area_filter_in),2))+", N="+str(len(area_filter_in)))
axs[0].hist(area_filter_all, bins, alpha=0.5, color="black")
axs[1].hist(area_filter_in, bins, alpha=0.5, color="black")
axs[0].set_ylabel('counts')
axs[1].set_xlabel('size (micrometers**2)')
axs[1].set_ylabel('counts')
fig.savefig(filename.replace(".geojson","")+"_size_hist.png")
fig.clf()

bins = np.linspace(0, 1, 100)

pyplot.hist(pp_filter_all, bins)
pyplot.title("Histogram of Values")
pyplot.xlabel("size (micrometers**2)")
pyplot.ylabel("counts") 
pyplot.savefig(filename.replace(".geojson","")+"_PP_hist.png")
pyplot.clf()


print("ALL:")
print("mean="+str(round(stats.mean(area_filter_all),2))+", med="+str(round(stats.median(area_filter_all),2))+", sd="+str(round(stats.stdev(area_filter_all),2))+", N="+str(len(area_filter_all)))

print("After filter:")
print("mean="+str(round(stats.mean(area_filter_in),2))+", med="+str(round(stats.median(area_filter_in),2))+", sd="+str(round(stats.stdev(area_filter_in),2))+", N="+str(len(area_filter_in)))


