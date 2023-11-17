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

leipzig_pixelSize = 0.5034
munich_pixelSize = 0.5034
hohenheim_pixelSize = 0.5034
gtex_pixelSize = 0.4942
endox_pixelSize = 0.2500

################################


dirName = "/gpfs3/well/lindgren/users/swf744/git/dev-happy/projects/adipose/trainingWithMunichLeipzigHohenheimV2_trainSamePixelSize_17feb/"
files = os.listdir(dirName)
leipzigFiles =  [f for f in files if f.startswith('a2') and not "0.5034" in f and "0.25" in f and not "_filtered" in f]

print("leipzigFiles")
print(leipzigFiles)
print(leipzigFiles[0])

leipzig_list = []
for i in leipzigFiles:
    print("i")
    print(i)
    res = readGeoJSON2list(dirName+i)
    leipzig_list.append(res)
    break

leipzig_res = [item for sublist in leipzig_list for item in sublist]



leipzig_res_area = []

for poly in leipzig_res:
        leipzig_res_area.append(poly.area*leipzig_pixelSize**2)


bins = np.linspace(0, 20000, 100)
pyplot.hist(leipzig_res_area, bins, alpha=0.5, color="black")
pyplot.title("size: mean="+str(round(stats.mean(leipzig_res_area),2))+", med="+str(round(stats.median(leipzig_res_area),2))+", sd="+str(round(stats.stdev(leipzig_res_area),2))+", N="+str(len(leipzig_res_area)))
pyplot.xlabel('size (micrometers**2)')
pyplot.ylabel('counts')
pyplot.savefig("happyPixelSize025_leipzig_hist_trainSamePixelSize_annoV2_noFilterSingleWSI_presentation23feb.png")
pyplot.clf()




leipzig_res_area2 = []

for poly in leipzig_res:
    if poly.area*leipzig_pixelSize**2 >= 200 and poly.area*leipzig_pixelSize**2 <= 16000 and ((4*math.pi*poly.area ) / ((poly.length)**2)) > 0.75:
        leipzig_res_area2.append(poly.area*leipzig_pixelSize**2)


bins = np.linspace(0, 20000, 100)
pyplot.hist(leipzig_res_area2, bins, alpha=0.5, color="black")
pyplot.title("size: mean="+str(round(stats.mean(leipzig_res_area2),2))+", med="+str(round(stats.median(leipzig_res_area2),2))+", sd="+str(round(stats.stdev(leipzig_res_area2),2))+", N="+str(len(leipzig_res_area2)))
pyplot.xlabel('size (micrometers**2)')
pyplot.ylabel('counts')
pyplot.savefig("happyPixelSize025_leipzig_hist_trainSamePixelSize_annoV2_filterSingleWSI_presentation23feb.png")
pyplot.clf()


#################################################################
###################################################################


dirName = "/gpfs3/well/lindgren/users/swf744/git/dev-happy/projects/adipose/trainingWithMunichLeipzigHohenheimV2_trainSamePixelSize_17feb/"
files = os.listdir(dirName)
leipzigFiles =  [f for f in files if f.startswith('a2') and not "0.5034" in f and "0.25" in f and not "_filtered" in f]

leipzig_list = []
for i in leipzigFiles:
    print("i")
    print(i)
    res = readGeoJSON2list(dirName+i)
    leipzig_list.append(res)

leipzig_res = [item for sublist in leipzig_list for item in sublist]



leipzig_res_area = []

for poly in leipzig_res:
        leipzig_res_area.append(poly.area*leipzig_pixelSize**2)


bins = np.linspace(0, 20000, 100)
pyplot.hist(leipzig_res_area, bins, alpha=0.5, color="black")
pyplot.title("size: mean="+str(round(stats.mean(leipzig_res_area),2))+", med="+str(round(stats.median(leipzig_res_area),2))+", sd="+str(round(stats.stdev(leipzig_res_area),2))+", N="+str(len(leipzig_res_area)))
pyplot.xlabel('size (micrometers**2)')
pyplot.ylabel('counts')
pyplot.savefig("happyPixelSize025_leipzig_hist_trainSamePixelSize_annoV2_noFilter_presentation23feb.png")
pyplot.clf()




leipzig_res_area2 = []

for poly in leipzig_res:
    if poly.area*leipzig_pixelSize**2 >= 200 and poly.area*leipzig_pixelSize**2 <= 16000 and ((4*math.pi*poly.area ) / ((poly.length)**2)) > 0.75:
        leipzig_res_area2.append(poly.area*leipzig_pixelSize**2)


bins = np.linspace(0, 20000, 100)
pyplot.hist(leipzig_res_area2, bins, alpha=0.5, color="black")
pyplot.title("size: mean="+str(round(stats.mean(leipzig_res_area2),2))+", med="+str(round(stats.median(leipzig_res_area2),2))+", sd="+str(round(stats.stdev(leipzig_res_area2),2))+", N="+str(len(leipzig_res_area2)))
pyplot.xlabel('size (micrometers**2)')
pyplot.ylabel('counts')
pyplot.savefig("happyPixelSize025_leipzig_hist_trainSamePixelSize_annoV2_filter_presentation23feb.png")
pyplot.clf()


