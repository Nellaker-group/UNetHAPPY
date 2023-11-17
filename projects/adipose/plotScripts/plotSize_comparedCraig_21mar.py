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

# Read CSV file into DataFrame df
df_sub = pd.read_csv('/gpfs3/well/lindgren/craig/isbi-2012/final_cohort_measurements/GTEx_Subcutaneous_areas.csv')
df_visc = pd.read_csv('/gpfs3/well/lindgren/craig/isbi-2012/final_cohort_measurements/GTEx_Visceral_areas.csv')


dirName = "/well/lindgren/users/swf744/git/dev-happy/projects/adipose/trainingWithMunichLeipzigHohenheimV2_testMultiInference_09mar/"
files = os.listdir(dirName)
gtexFiles =  [f for f in files if f.startswith('GTEX') and "0.25" in f ]


dirName2 = "/gpfs3/well/lindgren/users/swf744/git/dev-happy/projects/adipose/trainingWithMunichLeipzigHohenheimV2_testMultiInferenceMergedPolys_21mar/"
files2 = os.listdir(dirName2)
gtexFiles2 =  [f for f in files2 if f.startswith('GTEX') and "0.25" in f ]

sub_polys = {}
visc_polys = {}
for i in gtexFiles:
    print("i")
    print(i)
    res = readGeoJSON2list(dirName+i)
    if "Subcutaneous" in i:
        sub_polys[i.split("_")[0]] = res
    else:
        visc_polys[i.split("_")[0]] = res


sub_polys2 = {}
visc_polys2 = {}
for i in gtexFiles2:
    print("i")
    print(i)
    res = readGeoJSON2list(dirName2+i)
    if "Subcutaneous" in i:
        sub_polys2[i.split("_")[0]] = res
    else:
        visc_polys2[i.split("_")[0]] = res




sub_means = {}
visc_means = {}

for key in sub_polys.keys():
    area = 0
    counter = 0
    for poly in sub_polys[key]:
        if poly.area*gtex_pixelSize**2 >= 200 and poly.area*gtex_pixelSize**2 <= 16000 and ((4*math.pi*poly.area ) / ((poly.length)**2)) > 0.75:
            area += poly.area*gtex_pixelSize**2
            counter += 1
    sub_means[key] = area/counter


for key in visc_polys.keys():
    area = 0
    counter = 0
    for poly in visc_polys[key]:
        if poly.area*gtex_pixelSize**2 >= 200 and poly.area*gtex_pixelSize**2 <= 16000 and ((4*math.pi*poly.area ) / ((poly.length)**2)) > 0.75:
            area += poly.area*gtex_pixelSize**2
            counter += 1
    visc_means[key] = area/counter

sub_means2 = {}
visc_means2 = {}

for key in sub_polys2.keys():
    area = 0
    counter = 0
    for poly in sub_polys2[key]:
        if poly.area*gtex_pixelSize**2 >= 200 and poly.area*gtex_pixelSize**2 <= 16000 and ((4*math.pi*poly.area ) / ((poly.length)**2)) > 0.75:
            area += poly.area*gtex_pixelSize**2
            counter += 1
    sub_means2[key] = area/counter


for key in visc_polys2.keys():
    area = 0
    counter = 0
    for poly in visc_polys2[key]:
        if poly.area*gtex_pixelSize**2 >= 200 and poly.area*gtex_pixelSize**2 <= 16000 and ((4*math.pi*poly.area ) / ((poly.length)**2)) > 0.75:
            area += poly.area*gtex_pixelSize**2
            counter += 1
    visc_means2[key] = area/counter




plot_sub = []
plot_sub2 = []
plot_sub3 = []
for key in sub_means2.keys():
    if key in df_sub.SUBJID.values and key in sub_means.keys():
        plot_sub.append(sub_means[key])
        plot_sub2.append(sub_means2[key])
        plot_sub3.append(df_sub.loc[ df_sub["SUBJID"] == key,"mu_area"])


plot_visc = []
plot_visc2 = []
plot_visc3 = []
for key in visc_means2.keys():
    if key in df_visc.SUBJID.values and key in visc_means.keys():
        plot_visc.append(visc_means[key])
        plot_visc2.append(visc_means2[key])
        plot_visc3.append(df_visc.loc[ df_visc["SUBJID"] == key,"mu_area"])


x= []
for i in range(0,6000):
    x.append(i)



pyplot.scatter(plot_visc, plot_visc3,color="red",label="visc")
pyplot.scatter(plot_sub, plot_sub3,color="blue",label="sub")
pyplot.scatter(plot_visc2, plot_visc3,color="green",label="visc - merged pixel")
pyplot.scatter(plot_sub2, plot_sub3,color="yellow",label="sub - merged pixel")

pyplot.legend(loc="upper left")
pyplot.plot(x, x, "-", alpha=0.5, color="black",linewidth=2.0)
pyplot.title("GTEX mean size of adipocyte")
pyplot.xlabel('Emil')
pyplot.ylabel('Craig')
pyplot.ylim(1000, 5000)
pyplot.xlim(1000, 5000)
pyplot.savefig("plotSize_comparedCraig_21mar_gtex.png")
pyplot.clf()



