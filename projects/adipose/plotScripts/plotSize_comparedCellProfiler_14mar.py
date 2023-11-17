from shapely.geometry import Polygon
import pickle
import db.eval_runs_interface as db
from data.geojsoner import  writeToGeoJSON
import argparse
import math
from matplotlib import pyplot
import os
import statistics as stats
import numpy as np
import geojson
from shapely.geometry import Polygon, MultiPolygon, shape

# Load pandas
import pandas as pd

def readGeoJSON2list(filename):
    with open(filename) as f0:
        gj0 = geojson.load(f0)
    f0.close()
    polyList = geojson2polygon(gj0)
    return(polyList)

# little function for converting feature elements from geojson into shapely polygon objects
def geojson2polygon(gj):
    pols=[]
    for i in range(len(gj['features'])):
        pols.append(shape(gj['features'][i]["geometry"]))
    return(pols)

###########################################

gtex_pixelSize = 0.4942
munich_pixelSize = 0.5034

#########################################################


cp_munich_visc = pd.read_table('/gpfs3/well/lindgren/craig/Adipocyte-U-net/mobb_means.csv')


dirName = "/well/lindgren/users/swf744/git/dev-happy/projects/adipose/trainingWithMunichLeipzigHohenheimV2_testMultiInference_09mar/"
files = os.listdir(dirName)
munichFiles =  [f for f in files if f.startswith('m') and "0.25" in f and f.endswith("_px0.25.geojson")]

sub_polys = {}
visc_polys = {}

for i in munichFiles:
    print("i")
    print(i)
    res = readGeoJSON2list(dirName+i)
    if "sc" in i:
        sub_polys[i.split("_")[0]] = res
    else:
        visc_polys[i.split("_")[0]] = res



visc_means = {}
for key in visc_polys.keys():
    area = 0
    counter = 0
    for poly in visc_polys[key]:
        if poly.area*munich_pixelSize**2 >= 200 and poly.area*munich_pixelSize**2 <= 16000 and ((4*math.pi*poly.area ) / ((poly.length)**2)) > 0.75:
            area += poly.area*munich_pixelSize**2
            counter += 1
    print(key)
    print(area)
    print(counter)
    if counter > 0:
        visc_means[key] = area/counter
    else:
        visc_means[key] = 0



plot_visc = []
plot_visc2 = []
for key in visc_means.keys():
    key2 = key.split("v")[0]
    key2 = key2.split("s")[0]
    if key2 in cp_munich_visc.SUBJID.values:
        plot_visc.append(visc_means[key])
        plot_visc2.append(cp_munich_visc.loc[ cp_munich_visc["SUBJID"] == key2,"mu_area_vccp"])        
        print(key2)
        print(visc_means[key])
        print(cp_munich_visc.loc[ cp_munich_visc["SUBJID"] == key2,"mu_area_vccp"])



x= []
for i in range(0,6000):
    x.append(i)



pyplot.scatter(plot_visc, plot_visc2,color="red",label="visc")
pyplot.legend(loc="upper left")
pyplot.plot(x, x, "-", alpha=0.5, color="black",linewidth=2.0)
pyplot.title("MUNICH visceral mean size of adipocyte")
pyplot.xlabel('Emil')
pyplot.ylabel('Cell Profiler')
pyplot.ylim(1000, 5000)
pyplot.xlim(1000, 5000)
pyplot.savefig("plotSize_comparedCellprofiler_14mar_munich.png")
pyplot.clf()
