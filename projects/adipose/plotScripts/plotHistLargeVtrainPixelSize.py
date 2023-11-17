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

dirName = "/gpfs3/well/lindgren/users/swf744/git/dev-happy/projects/adipose/trainingWithMunichLeipzigHohenheim_HappyLargerPixelSize/"
files = os.listdir(dirName)
leipzigFiles =  [f for f in files if f.startswith('a2')]

leipzig_list = []
for i in leipzigFiles:
    print("i")
    print(i)
    res = readGeoJSON2list(dirName+i)
    leipzig_list.append(res)

leipzig_res = [item for sublist in leipzig_list for item in sublist]



leipzig_res_area = []

for poly in leipzig_res:
    if poly.area*leipzig_pixelSize**2 >= 200 and poly.area*leipzig_pixelSize**2 <= 16000 and ((4*math.pi*poly.area ) / ((poly.length)**2)) > 0.75:        
        leipzig_res_area.append(poly.area*leipzig_pixelSize**2)


bins = np.linspace(200, 16000, 100)
pyplot.hist(leipzig_res_area, bins, alpha=0.5, color="black")
pyplot.title("size: mean="+str(round(stats.mean(leipzig_res_area),2))+", med="+str(round(stats.median(leipzig_res_area),2))+", sd="+str(round(stats.stdev(leipzig_res_area),2))+", N="+str(len(leipzig_res_area)))
pyplot.xlabel('size (micrometers**2)')
pyplot.ylabel('counts')
pyplot.savefig("happyLargerPixelSize_leipzig_hist.png")
pyplot.clf()

#########################################################


dirName = "/gpfs3/well/lindgren/users/swf744/git/dev-happy/projects/adipose/trainingWithMunichLeipzigHohenheim_HappyTrainPixelSize/"
files = os.listdir(dirName)
leipzigFiles =  [f for f in files if f.startswith('a2')]

leipzig_list = []
for i in leipzigFiles:
    print("i")
    print(i)
    res = readGeoJSON2list(dirName+i)
    leipzig_list.append(res)

leipzig_res = [item for sublist in leipzig_list for item in sublist]



leipzig_res_area = []

for poly in leipzig_res:
    if poly.area*leipzig_pixelSize**2 >= 200 and poly.area*leipzig_pixelSize**2 <= 16000 and ((4*math.pi*poly.area ) / ((poly.length)**2)) > 0.75:        
        leipzig_res_area.append(poly.area*leipzig_pixelSize**2)


bins = np.linspace(200, 16000, 100)
pyplot.hist(leipzig_res_area, bins, alpha=0.5, color="black")
pyplot.title("size: mean="+str(round(stats.mean(leipzig_res_area),2))+", med="+str(round(stats.median(leipzig_res_area),2))+", sd="+str(round(stats.stdev(leipzig_res_area),2))+", N="+str(len(leipzig_res_area)))
pyplot.xlabel('size (micrometers**2)')
pyplot.ylabel('counts')
pyplot.savefig("happyTrainPixelSize_leipzig_hist.png")
pyplot.clf()

###########################################################

dirName = "/gpfs3/well/lindgren/users/swf744/git/dev-happy/projects/adipose/trainingWithMunichLeipzigHohenheim_HappyLargerPixelSize/"
files = os.listdir(dirName)
munichFiles =  [f for f in files if f.startswith('m')]

munich_list = []
for i in munichFiles:
    print("i")
    print(i)
    res = readGeoJSON2list(dirName+i)
    munich_list.append(res)

munich_res = [item for sublist in munich_list for item in sublist]



munich_res_area = []

for poly in munich_res:
    if poly.area*munich_pixelSize**2 >= 200 and poly.area*munich_pixelSize**2 <= 16000 and ((4*math.pi*poly.area ) / ((poly.length)**2)) > 0.75:        
        munich_res_area.append(poly.area*munich_pixelSize**2)


bins = np.linspace(200, 16000, 100)
pyplot.hist(munich_res_area, bins, alpha=0.5, color="black")
pyplot.title("size: mean="+str(round(stats.mean(munich_res_area),2))+", med="+str(round(stats.median(munich_res_area),2))+", sd="+str(round(stats.stdev(munich_res_area),2))+", N="+str(len(munich_res_area)))
pyplot.xlabel('size (micrometers**2)')
pyplot.ylabel('counts')
pyplot.savefig("happyLargerPixelSize_munich_hist.png")
pyplot.clf()

#########################################################


dirName = "/gpfs3/well/lindgren/users/swf744/git/dev-happy/projects/adipose/trainingWithMunichLeipzigHohenheim_HappyTrainPixelSize/"
files = os.listdir(dirName)
munichFiles =  [f for f in files if f.startswith('m')]

munich_list = []
for i in munichFiles:
    print("i")
    print(i)
    res = readGeoJSON2list(dirName+i)
    munich_list.append(res)

munich_res = [item for sublist in munich_list for item in sublist]



munich_res_area = []

for poly in munich_res:
    if poly.area*munich_pixelSize**2 >= 200 and poly.area*munich_pixelSize**2 <= 16000 and ((4*math.pi*poly.area ) / ((poly.length)**2)) > 0.75:        
        munich_res_area.append(poly.area*munich_pixelSize**2)


bins = np.linspace(200, 16000, 100)
pyplot.hist(munich_res_area, bins, alpha=0.5, color="black")
pyplot.title("size: mean="+str(round(stats.mean(munich_res_area),2))+", med="+str(round(stats.median(munich_res_area),2))+", sd="+str(round(stats.stdev(munich_res_area),2))+", N="+str(len(munich_res_area)))
pyplot.xlabel('size (micrometers**2)')
pyplot.ylabel('counts')
pyplot.savefig("happyTrainPixelSize_munich_hist.png")
pyplot.clf()


###########################################################

dirName = "/gpfs3/well/lindgren/users/swf744/git/dev-happy/projects/adipose/trainingWithMunichLeipzigHohenheim_HappyLargerPixelSize/"
files = os.listdir(dirName)
hohenheimFiles =  [f for f in files if f.startswith('h')]

hohenheim_list = []
for i in hohenheimFiles:
    print("i")
    print(i)
    res = readGeoJSON2list(dirName+i)
    hohenheim_list.append(res)

hohenheim_res = [item for sublist in hohenheim_list for item in sublist]



hohenheim_res_area = []

for poly in hohenheim_res:
    if poly.area*hohenheim_pixelSize**2 >= 200 and poly.area*hohenheim_pixelSize**2 <= 16000 and ((4*math.pi*poly.area ) / ((poly.length)**2)) > 0.75:        
        hohenheim_res_area.append(poly.area*hohenheim_pixelSize**2)


bins = np.linspace(200, 16000, 100)
pyplot.hist(hohenheim_res_area, bins, alpha=0.5, color="black")
pyplot.title("size: mean="+str(round(stats.mean(hohenheim_res_area),2))+", med="+str(round(stats.median(hohenheim_res_area),2))+", sd="+str(round(stats.stdev(hohenheim_res_area),2))+", N="+str(len(hohenheim_res_area)))
pyplot.xlabel('size (micrometers**2)')
pyplot.ylabel('counts')
pyplot.savefig("happyLargerPixelSize_hohenheim_hist.png")
pyplot.clf()

#########################################################


dirName = "/gpfs3/well/lindgren/users/swf744/git/dev-happy/projects/adipose/trainingWithMunichLeipzigHohenheim_HappyTrainPixelSize/"
files = os.listdir(dirName)
hohenheimFiles =  [f for f in files if f.startswith('h')]

hohenheim_list = []
for i in hohenheimFiles:
    print("i")
    print(i)
    res = readGeoJSON2list(dirName+i)
    hohenheim_list.append(res)

hohenheim_res = [item for sublist in hohenheim_list for item in sublist]



hohenheim_res_area = []

for poly in hohenheim_res:
    if poly.area*hohenheim_pixelSize**2 >= 200 and poly.area*hohenheim_pixelSize**2 <= 16000 and ((4*math.pi*poly.area ) / ((poly.length)**2)) > 0.75:        
        hohenheim_res_area.append(poly.area*hohenheim_pixelSize**2)


bins = np.linspace(200, 16000, 100)
pyplot.hist(hohenheim_res_area, bins, alpha=0.5, color="black")
pyplot.title("size: mean="+str(round(stats.mean(hohenheim_res_area),2))+", med="+str(round(stats.median(hohenheim_res_area),2))+", sd="+str(round(stats.stdev(hohenheim_res_area),2))+", N="+str(len(hohenheim_res_area)))
pyplot.xlabel('size (micrometers**2)')
pyplot.ylabel('counts')
pyplot.savefig("happyTrainPixelSize_hohenheim_hist.png")
pyplot.clf()


