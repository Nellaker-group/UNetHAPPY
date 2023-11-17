from shapely.geometry import Polygon
import pickle
import sys
import argparse
import math
from matplotlib import pyplot
import os
import statistics as stats
import numpy as np
import geojson
from shapely.geometry import Polygon, MultiPolygon, shape

sys.path.append(os.getcwd())
import db.eval_runs_interface as db
from data.geojsoner import  writeToGeoJSON

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

dirName = "/well/lindgren/users/swf744/git/dev-happy/projects/adipose/trainingWithMunichLeipzigHohenheimV2_testMultiInference_09mar/"
files = os.listdir(dirName)
leipzigFiles =  [f for f in files if f.startswith('a') and "0.25" in f and "vc" in f and f.endswith("_px0.25.geojson")]

print(leipzigFiles)

leipzig_list = []
for i in leipzigFiles:
    print("i")
    print(i)
    res = readGeoJSON2list(dirName+i)
    leipzig_list.append(res)

leipzig_res = [item for sublist in leipzig_list for item in sublist]



leipzig_res_diam = []

for poly in leipzig_res:
    if poly.area*leipzig_pixelSize**2 >= 200 and poly.area*leipzig_pixelSize**2 <= 16000 and ((4*math.pi*poly.area ) / ((poly.length)**2)) > 0.75:
        leipzig_res_diam.append(2*math.sqrt(poly.area*leipzig_pixelSize**2/math.pi))


bins = np.linspace(15, 150, 100)
pyplot.hist(leipzig_res_diam, bins, alpha=0.5, color="black")
pyplot.title("size: mean="+str(round(stats.mean(leipzig_res_diam),2))+", med="+str(round(stats.median(leipzig_res_diam),2))+", sd="+str(round(stats.stdev(leipzig_res_diam),2))+", N="+str(len(leipzig_res_diam)))
pyplot.xlabel('diam (micrometers)')
pyplot.ylabel('counts')
pyplot.savefig("leipzig_histDiameter_annoV2_px025_visc_testMultiInference_22mar.png")
pyplot.clf()



################################


dirName = "/well/lindgren/users/swf744/git/dev-happy/projects/adipose/trainingWithMunichLeipzigHohenheimV2_testMultiInference_09mar/"
files = os.listdir(dirName)
leipzigFiles =  [f for f in files if f.startswith('a') and "0.25" in f and "sc" in f and f.endswith("_px0.25.geojson")]

leipzig_list = []
for i in leipzigFiles:
    print("i")
    print(i)
    res = readGeoJSON2list(dirName+i)
    leipzig_list.append(res)

leipzig_res = [item for sublist in leipzig_list for item in sublist]



leipzig_res_diam = []

for poly in leipzig_res:
    if poly.area*leipzig_pixelSize**2 >= 200 and poly.area*leipzig_pixelSize**2 <= 16000 and ((4*math.pi*poly.area ) / ((poly.length)**2)) > 0.75:
        leipzig_res_diam.append(2*math.sqrt(poly.area*leipzig_pixelSize**2/math.pi))


bins = np.linspace(15, 150, 100)
pyplot.hist(leipzig_res_diam, bins, alpha=0.5, color="black")
pyplot.title("size: mean="+str(round(stats.mean(leipzig_res_diam),2))+", med="+str(round(stats.median(leipzig_res_diam),2))+", sd="+str(round(stats.stdev(leipzig_res_diam),2))+", N="+str(len(leipzig_res_diam)))
pyplot.xlabel('diam (micrometers)')
pyplot.ylabel('counts')
pyplot.savefig("leipzig_histDiameter_annoV2_px025_subc_testMultiInference_22mar.png")
pyplot.clf()

################################

dirName = "/well/lindgren/users/swf744/git/dev-happy/projects/adipose/trainingWithMunichLeipzigHohenheimV2_testMultiInference_09mar/"
files = os.listdir(dirName)
munichFiles =  [f for f in files if f.startswith('m') and "0.25" in f and "vc" in f and f.endswith("_px0.25.geojson")]

munich_list = []
for i in munichFiles:
    print("i")
    print(i)
    res = readGeoJSON2list(dirName+i)
    munich_list.append(res)

munich_res = [item for sublist in munich_list for item in sublist]



munich_res_diam = []

for poly in munich_res:
    if poly.area*munich_pixelSize**2 >= 200 and poly.area*munich_pixelSize**2 <= 16000 and ((4*math.pi*poly.area ) / ((poly.length)**2)) > 0.75:
        munich_res_diam.append(2*math.sqrt(poly.area*munich_pixelSize**2/math.pi))


bins = np.linspace(15, 150, 100)
pyplot.hist(munich_res_diam, bins, alpha=0.5, color="black")
pyplot.title("size: mean="+str(round(stats.mean(munich_res_diam),2))+", med="+str(round(stats.median(munich_res_diam),2))+", sd="+str(round(stats.stdev(munich_res_diam),2))+", N="+str(len(munich_res_diam)))
pyplot.xlabel('diam (micrometers)')
pyplot.ylabel('counts')
pyplot.savefig("munich_hist_annoV2_px025_visc_testMultiInference_22mar.png")
pyplot.clf()

#########################################################


dirName = "/well/lindgren/users/swf744/git/dev-happy/projects/adipose/trainingWithMunichLeipzigHohenheimV2_testMultiInference_09mar/"
files = os.listdir(dirName)
munichFiles =  [f for f in files if f.startswith('m') and "0.25" in f and "sc" in f and f.endswith("_px0.25.geojson")]

munich_list = []
for i in munichFiles:
    print("i")
    print(i)
    res = readGeoJSON2list(dirName+i)
    munich_list.append(res)

munich_res = [item for sublist in munich_list for item in sublist]



munich_res_diam = []

for poly in munich_res:
    if poly.area*munich_pixelSize**2 >= 200 and poly.area*munich_pixelSize**2 <= 16000 and ((4*math.pi*poly.area ) / ((poly.length)**2)) > 0.75:
        munich_res_diam.append(2*math.sqrt(poly.area*munich_pixelSize**2/math.pi))


bins = np.linspace(15, 150, 100)
pyplot.hist(munich_res_diam, bins, alpha=0.5, color="black")
pyplot.title("size: mean="+str(round(stats.mean(munich_res_diam),2))+", med="+str(round(stats.median(munich_res_diam),2))+", sd="+str(round(stats.stdev(munich_res_diam),2))+", N="+str(len(munich_res_diam)))
pyplot.xlabel('diam (micrometers)')
pyplot.ylabel('counts')
pyplot.savefig("munich_histDiameter_annoV2_px025_subc_testMultiInference_22mar.png")
pyplot.clf()

#########################################################


dirName = "/well/lindgren/users/swf744/git/dev-happy/projects/adipose/trainingWithMunichLeipzigHohenheimV2_testMultiInference_09mar/"
files = os.listdir(dirName)
hohenheimFiles =  [f for f in files if f.startswith('h') and "0.25" in f and "vc" in f and f.endswith("_px0.25.geojson")]

hohenheim_list = []
for i in hohenheimFiles:
    print("i")
    print(i)
    res = readGeoJSON2list(dirName+i)
    hohenheim_list.append(res)

hohenheim_res = [item for sublist in hohenheim_list for item in sublist]



hohenheim_res_diam = []

for poly in hohenheim_res:
    if poly.area*hohenheim_pixelSize**2 >= 200 and poly.area*hohenheim_pixelSize**2 <= 16000 and ((4*math.pi*poly.area ) / ((poly.length)**2)) > 0.75:
        hohenheim_res_diam.append(2*math.sqrt(poly.area*hohenheim_pixelSize**2/math.pi))


bins = np.linspace(15, 150, 100)
pyplot.hist(hohenheim_res_diam, bins, alpha=0.5, color="black")
pyplot.title("size: mean="+str(round(stats.mean(hohenheim_res_diam),2))+", med="+str(round(stats.median(hohenheim_res_diam),2))+", sd="+str(round(stats.stdev(hohenheim_res_diam),2))+", N="+str(len(hohenheim_res_diam)))
pyplot.xlabel('diam (micrometers)')
pyplot.ylabel('counts')
pyplot.savefig("hohenheim_histDiameter_annoV2_px025_visc_testMultiInference_22mar.png")
pyplot.clf()

#########################################################


dirName = "/well/lindgren/users/swf744/git/dev-happy/projects/adipose/trainingWithMunichLeipzigHohenheimV2_testMultiInference_09mar/"
files = os.listdir(dirName)
hohenheimFiles =  [f for f in files if f.startswith('h') and "0.25" in f and "sc" in f and f.endswith("_px0.25.geojson")]

hohenheim_list = []
for i in hohenheimFiles:
    print("i")
    print(i)
    res = readGeoJSON2list(dirName+i)
    hohenheim_list.append(res)

hohenheim_res = [item for sublist in hohenheim_list for item in sublist]



hohenheim_res_diam = []

for poly in hohenheim_res:
    if poly.area*hohenheim_pixelSize**2 >= 200 and poly.area*hohenheim_pixelSize**2 <= 16000 and ((4*math.pi*poly.area ) / ((poly.length)**2)) > 0.75:
        hohenheim_res_diam.append(2*math.sqrt(poly.area*hohenheim_pixelSize**2/math.pi))


bins = np.linspace(15, 150, 100)
pyplot.hist(hohenheim_res_diam, bins, alpha=0.5, color="black")
pyplot.title("size: mean="+str(round(stats.mean(hohenheim_res_diam),2))+", med="+str(round(stats.median(hohenheim_res_diam),2))+", sd="+str(round(stats.stdev(hohenheim_res_diam),2))+", N="+str(len(hohenheim_res_diam)))
pyplot.xlabel('diam (micrometers)')
pyplot.ylabel('counts')
pyplot.savefig("hohenheim_histDiameter_annoV2_px025_subc_testMultiInference_22mar.png")
pyplot.clf()


#########################################################


dirName = "/well/lindgren/users/swf744/git/dev-happy/projects/adipose/trainingWithMunichLeipzigHohenheimV2_testMultiInference_09mar/"
files = os.listdir(dirName)
endoxFiles =  [f for f in files if f.startswith('Image')  and "0.25" in f and f.endswith("_px0.25.geojson")]

endox_list = []
for i in endoxFiles:
    print("i")
    print(i)
    res = readGeoJSON2list(dirName+i)
    endox_list.append(res)

endox_res = [item for sublist in endox_list for item in sublist]



endox_res_diam = []

for poly in endox_res:
    if poly.area*endox_pixelSize**2 >= 200 and poly.area*endox_pixelSize**2 <= 16000 and ((4*math.pi*poly.area ) / ((poly.length)**2)) > 0.75:
        endox_res_diam.append(2*math.sqrt(poly.area*endox_pixelSize**2/math.pi))


bins = np.linspace(15, 150, 100)
pyplot.hist(endox_res_diam, bins, alpha=0.5, color="black")
pyplot.title("size: mean="+str(round(stats.mean(endox_res_diam),2))+", med="+str(round(stats.median(endox_res_diam),2))+", sd="+str(round(stats.stdev(endox_res_diam),2))+", N="+str(len(endox_res_diam)))
pyplot.xlabel('diam (micrometers)')
pyplot.ylabel('counts')
pyplot.savefig("endox_histDiameter_annoV2_px025_testMultiInference_22mar.png")
pyplot.clf()

#########################################################


dirName = "/well/lindgren/users/swf744/git/dev-happy/projects/adipose/trainingWithMunichLeipzigHohenheimV2_testMultiInference_09mar/"
files = os.listdir(dirName)
gtexFiles =  [f for f in files if f.startswith('GTEX') and "0.25" in f and "Visceral" in f  and f.endswith("_px0.25.geojson")]

gtex_list = []
for i in gtexFiles:
    print("i")
    print(i)
    res = readGeoJSON2list(dirName+i)
    gtex_list.append(res)

gtex_res = [item for sublist in gtex_list for item in sublist]



gtex_res_diam = []

for poly in gtex_res:
    if poly.area*gtex_pixelSize**2 >= 200 and poly.area*gtex_pixelSize**2 <= 16000 and ((4*math.pi*poly.area ) / ((poly.length)**2)) > 0.75:
        gtex_res_diam.append(2*math.sqrt(poly.area*gtex_pixelSize**2/math.pi))


bins = np.linspace(15, 150, 100)
pyplot.hist(gtex_res_diam, bins, alpha=0.5, color="black")
pyplot.title("size: mean="+str(round(stats.mean(gtex_res_diam),2))+", med="+str(round(stats.median(gtex_res_diam),2))+", sd="+str(round(stats.stdev(gtex_res_diam),2))+", N="+str(len(gtex_res_diam)))
pyplot.xlabel('diam (micrometers)')
pyplot.ylabel('counts')
pyplot.savefig("gtex_histDiameter_annoV2_px025_visc_testMultiInference_22mar.png")
pyplot.clf()



#########################################################


dirName = "/well/lindgren/users/swf744/git/dev-happy/projects/adipose/trainingWithMunichLeipzigHohenheimV2_testMultiInference_09mar/"
files = os.listdir(dirName)
gtexFiles =  [f for f in files if f.startswith('GTEX') and "0.25" in f and "Subcutaneous" in f  and f.endswith("_px0.25.geojson")]

gtex_list = []
for i in gtexFiles:
    print("i")
    print(i)
    res = readGeoJSON2list(dirName+i)
    gtex_list.append(res)

gtex_res = [item for sublist in gtex_list for item in sublist]



gtex_res_diam = []

for poly in gtex_res:
    if poly.area*gtex_pixelSize**2 >= 200 and poly.area*gtex_pixelSize**2 <= 16000 and ((4*math.pi*poly.area ) / ((poly.length)**2)) > 0.75:
        gtex_res_diam.append(2*math.sqrt(poly.area*gtex_pixelSize**2/math.pi))


bins = np.linspace(15, 150, 100)
pyplot.hist(gtex_res_diam, bins, alpha=0.5, color="black")
pyplot.title("size: mean="+str(round(stats.mean(gtex_res_diam),2))+", med="+str(round(stats.median(gtex_res_diam),2))+", sd="+str(round(stats.stdev(gtex_res_diam),2))+", N="+str(len(gtex_res_diam)))
pyplot.xlabel('diam (micrometers)')
pyplot.ylabel('counts')
pyplot.savefig("gtex_histDiameter_annoV2_px025_subc_testMultiInference_22mar.png")
pyplot.clf()



