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

dirName = "/well/lindgren/users/swf744/git/dev-happy/projects/adipose/trainingWithMunichLeipzigHohenheimV2_testMultiInferenceMergedPolys_21mar/"
dirName2 = "/well/lindgren/users/swf744/git/dev-happy/projects/adipose/trainingWithMunichLeipzigHohenheimV2_testMultiInference_09mar/"
files = os.listdir(dirName)
leipzigFiles =  [f for f in files if f.startswith('a') and "0.25" in f and "vc" in f and f.endswith('.geojson')]

leipzig_list = []
leipzig_list2 = []
for i in leipzigFiles:
    print("i")
    print(i)
    res = readGeoJSON2list(dirName+i)
    leipzig_list.append(res)
    res2 = readGeoJSON2list(dirName2+i.replace("_px1_0.25_px2_0.5034","_px0.25").replace(".svs_merged","coords_merged_segmodel1_overlap256"))
    leipzig_list2.append(res2)



leipzig_res = [item for sublist in leipzig_list for item in sublist]
leipzig_res2 = [item for sublist in leipzig_list2 for item in sublist]

leipzig_res_area = []
leipzig_res_area2 = []

for poly in leipzig_res:
    if poly.area*leipzig_pixelSize**2 >= 200 and poly.area*leipzig_pixelSize**2 <= 16000 and ((4*math.pi*poly.area ) / ((poly.length)**2)) > 0.75:
        leipzig_res_area.append(poly.area*leipzig_pixelSize**2)
for poly in leipzig_res2:
    if poly.area*leipzig_pixelSize**2 >= 200 and poly.area*leipzig_pixelSize**2 <= 16000 and ((4*math.pi*poly.area ) / ((poly.length)**2)) > 0.75:
        leipzig_res_area2.append(poly.area*leipzig_pixelSize**2)

bins = np.linspace(200, 16000, 100)

fig, axs = pyplot.subplots(2)
fig.suptitle("merged-size: mean="+str(round(stats.mean(leipzig_res_area),2))+", med="+str(round(stats.median(leipzig_res_area),2))+", sd="+str(round(stats.stdev(leipzig_res_area),2))+", N="+str(len(leipzig_res_area))+"\n"
             "size: mean="+str(round(stats.mean(leipzig_res_area2),2))+", med="+str(round(stats.median(leipzig_res_area2),2))+", sd="+str(round(stats.stdev(leipzig_res_area2),2))+", N="+str(len(leipzig_res_area2)))
axs[0].hist(leipzig_res_area, bins, alpha=0.5, color="black")
axs[1].hist(leipzig_res_area2, bins, alpha=0.5, color="black")
axs[0].set_ylabel('counts')
axs[1].set_xlabel('size (micrometers**2)')
axs[1].set_ylabel('counts')
fig.savefig("leipzig_hist_annoV2_px025_px05034_visc_testMultiInferenceMergedPolys_v_nonMerged_23mar.png")
fig.clf()

################################

dirName = "/well/lindgren/users/swf744/git/dev-happy/projects/adipose/trainingWithMunichLeipzigHohenheimV2_testMultiInferenceMergedPolys_21mar/"
dirName2 = "/well/lindgren/users/swf744/git/dev-happy/projects/adipose/trainingWithMunichLeipzigHohenheimV2_testMultiInference_09mar/"
files = os.listdir(dirName)
leipzigFiles =  [f for f in files if f.startswith('a') and "0.25" in f and "sc" in f and f.endswith('.geojson')]

leipzig_list = []
leipzig_list2 = []
for i in leipzigFiles:
    print("i")
    print(i)
    res = readGeoJSON2list(dirName+i)
    leipzig_list.append(res)
    res2 = readGeoJSON2list(dirName2+i.replace("_px1_0.25_px2_0.5034","_px0.25").replace(".svs_merged","coords_merged_segmodel1_overlap256"))
    leipzig_list2.append(res2)


leipzig_res = [item for sublist in leipzig_list for item in sublist]
leipzig_res2 = [item for sublist in leipzig_list2 for item in sublist]

leipzig_res_area = []
leipzig_res_area2 = []

for poly in leipzig_res:
    if poly.area*leipzig_pixelSize**2 >= 200 and poly.area*leipzig_pixelSize**2 <= 16000 and ((4*math.pi*poly.area ) / ((poly.length)**2)) > 0.75:
        leipzig_res_area.append(poly.area*leipzig_pixelSize**2)
for poly in leipzig_res2:
    if poly.area*leipzig_pixelSize**2 >= 200 and poly.area*leipzig_pixelSize**2 <= 16000 and ((4*math.pi*poly.area ) / ((poly.length)**2)) > 0.75:
        leipzig_res_area2.append(poly.area*leipzig_pixelSize**2)

bins = np.linspace(200, 16000, 100)

fig, axs = pyplot.subplots(2)
fig.suptitle("merged-size: mean="+str(round(stats.mean(leipzig_res_area),2))+", med="+str(round(stats.median(leipzig_res_area),2))+", sd="+str(round(stats.stdev(leipzig_res_area),2))+", N="+str(len(leipzig_res_area))+"\n"
             "size: mean="+str(round(stats.mean(leipzig_res_area2),2))+", med="+str(round(stats.median(leipzig_res_area2),2))+", sd="+str(round(stats.stdev(leipzig_res_area2),2))+", N="+str(len(leipzig_res_area2)))
axs[0].hist(leipzig_res_area, bins, alpha=0.5, color="black")
axs[1].hist(leipzig_res_area2, bins, alpha=0.5, color="black")
axs[0].set_ylabel('counts')
axs[1].set_xlabel('size (micrometers**2)')
axs[1].set_ylabel('counts')
fig.savefig("leipzig_hist_annoV2_px025_px05034_subc_testMultiInferenceMergedPolys_v_nonMerged_23mar.png")
fig.clf()


###############################


dirName = "/well/lindgren/users/swf744/git/dev-happy/projects/adipose/trainingWithMunichLeipzigHohenheimV2_testMultiInferenceMergedPolys_21mar/"
dirName2 = "/well/lindgren/users/swf744/git/dev-happy/projects/adipose/trainingWithMunichLeipzigHohenheimV2_testMultiInference_09mar/"
files = os.listdir(dirName)
munichFiles =  [f for f in files if f.startswith('m') and "0.25" in f and "vc" in f and f.endswith('.geojson')]

munich_list = []
munich_list2 = []
for i in munichFiles:
    print("i")
    print(i)
    res = readGeoJSON2list(dirName+i)
    munich_list.append(res)
    res2 = readGeoJSON2list(dirName2+i.replace("_px1_0.25_px2_0.5034","_px0.25").replace(".svs_merged","coords_merged_segmodel1_overlap256"))
    munich_list2.append(res2)


munich_res = [item for sublist in munich_list for item in sublist]
munich_res2 = [item for sublist in munich_list2 for item in sublist]

munich_res_area = []
munich_res_area2 = []

for poly in munich_res:
    if poly.area*munich_pixelSize**2 >= 200 and poly.area*munich_pixelSize**2 <= 16000 and ((4*math.pi*poly.area ) / ((poly.length)**2)) > 0.75:
        munich_res_area.append(poly.area*munich_pixelSize**2)
for poly in munich_res2:
    if poly.area*munich_pixelSize**2 >= 200 and poly.area*munich_pixelSize**2 <= 16000 and ((4*math.pi*poly.area ) / ((poly.length)**2)) > 0.75:
        munich_res_area2.append(poly.area*munich_pixelSize**2)

bins = np.linspace(200, 16000, 100)

fig, axs = pyplot.subplots(2)
fig.suptitle("merged-size: mean="+str(round(stats.mean(munich_res_area),2))+", med="+str(round(stats.median(munich_res_area),2))+", sd="+str(round(stats.stdev(munich_res_area),2))+", N="+str(len(munich_res_area))+"\n"
             "size: mean="+str(round(stats.mean(munich_res_area2),2))+", med="+str(round(stats.median(munich_res_area2),2))+", sd="+str(round(stats.stdev(munich_res_area2),2))+", N="+str(len(munich_res_area2)))
axs[0].hist(munich_res_area, bins, alpha=0.5, color="black")
axs[1].hist(munich_res_area2, bins, alpha=0.5, color="black")
axs[0].set_ylabel('counts')
axs[1].set_xlabel('size (micrometers**2)')
axs[1].set_ylabel('counts')
fig.savefig("munich_hist_annoV2_px025_px05034_visc_testMultiInferenceMergedPolys_v_nonMerged_23mar.png")
fig.clf()

#########################################################


dirName = "/well/lindgren/users/swf744/git/dev-happy/projects/adipose/trainingWithMunichLeipzigHohenheimV2_testMultiInferenceMergedPolys_21mar/"
dirName2 = "/well/lindgren/users/swf744/git/dev-happy/projects/adipose/trainingWithMunichLeipzigHohenheimV2_testMultiInference_09mar/"
files = os.listdir(dirName)
munichFiles =  [f for f in files if f.startswith('m') and "0.25" in f and "sc" in f and f.endswith('.geojson')]

munich_list = []
munich_list2 = []
for i in munichFiles:
    print("i")
    print(i)
    res = readGeoJSON2list(dirName+i)
    munich_list.append(res)
    res2 = readGeoJSON2list(dirName2+i.replace("_px1_0.25_px2_0.5034","_px0.25").replace(".svs_merged","coords_merged_segmodel1_overlap256"))
    munich_list2.append(res2)


munich_res = [item for sublist in munich_list for item in sublist]
munich_res2 = [item for sublist in munich_list2 for item in sublist]

munich_res_area = []
munich_res_area2 = []

for poly in munich_res:
    if poly.area*munich_pixelSize**2 >= 200 and poly.area*munich_pixelSize**2 <= 16000 and ((4*math.pi*poly.area ) / ((poly.length)**2)) > 0.75:
        munich_res_area.append(poly.area*munich_pixelSize**2)
for poly in munich_res2:
    if poly.area*munich_pixelSize**2 >= 200 and poly.area*munich_pixelSize**2 <= 16000 and ((4*math.pi*poly.area ) / ((poly.length)**2)) > 0.75:
        munich_res_area2.append(poly.area*munich_pixelSize**2)

bins = np.linspace(200, 16000, 100)

fig, axs = pyplot.subplots(2)
fig.suptitle("merged-size: mean="+str(round(stats.mean(munich_res_area),2))+", med="+str(round(stats.median(munich_res_area),2))+", sd="+str(round(stats.stdev(munich_res_area),2))+", N="+str(len(munich_res_area))+"\n"
             "size: mean="+str(round(stats.mean(munich_res_area2),2))+", med="+str(round(stats.median(munich_res_area2),2))+", sd="+str(round(stats.stdev(munich_res_area2),2))+", N="+str(len(munich_res_area2)))
axs[0].hist(munich_res_area, bins, alpha=0.5, color="black")
axs[1].hist(munich_res_area2, bins, alpha=0.5, color="black")
axs[0].set_ylabel('counts')
axs[1].set_xlabel('size (micrometers**2)')
axs[1].set_ylabel('counts')
fig.savefig("munich_hist_annoV2_px025_px05034_subc_testMultiInferenceMergedPolys_v_nonMerged_23mar.png")
fig.clf()


##########################################################


dirName = "/well/lindgren/users/swf744/git/dev-happy/projects/adipose/trainingWithMunichLeipzigHohenheimV2_testMultiInferenceMergedPolys_21mar/"
dirName2 = "/well/lindgren/users/swf744/git/dev-happy/projects/adipose/trainingWithMunichLeipzigHohenheimV2_testMultiInference_09mar/"
files = os.listdir(dirName)
hohenheimFiles =  [f for f in files if f.startswith('h') and "0.25" in f and "vc" in f and f.endswith('.geojson')]

hohenheim_list = []
hohenheim_list2 = []
for i in hohenheimFiles:
    print("i")
    print(i)
    res = readGeoJSON2list(dirName+i)
    hohenheim_list.append(res)
    res2 = readGeoJSON2list(dirName2+i.replace("_px1_0.25_px2_0.5034","_px0.25").replace(".svs_merged","coords_merged_segmodel1_overlap256"))
    hohenheim_list2.append(res2)


hohenheim_res = [item for sublist in hohenheim_list for item in sublist]
hohenheim_res2 = [item for sublist in hohenheim_list2 for item in sublist]

hohenheim_res_area = []
hohenheim_res_area2 = []

for poly in hohenheim_res:
    if poly.area*hohenheim_pixelSize**2 >= 200 and poly.area*hohenheim_pixelSize**2 <= 16000 and ((4*math.pi*poly.area ) / ((poly.length)**2)) > 0.75:
        hohenheim_res_area.append(poly.area*hohenheim_pixelSize**2)
for poly in hohenheim_res2:
    if poly.area*hohenheim_pixelSize**2 >= 200 and poly.area*hohenheim_pixelSize**2 <= 16000 and ((4*math.pi*poly.area ) / ((poly.length)**2)) > 0.75:
        hohenheim_res_area2.append(poly.area*hohenheim_pixelSize**2)

bins = np.linspace(200, 16000, 100)

fig, axs = pyplot.subplots(2)
fig.suptitle("merged-size: mean="+str(round(stats.mean(hohenheim_res_area),2))+", med="+str(round(stats.median(hohenheim_res_area),2))+", sd="+str(round(stats.stdev(hohenheim_res_area),2))+", N="+str(len(hohenheim_res_area))+"\n"
             "size: mean="+str(round(stats.mean(hohenheim_res_area2),2))+", med="+str(round(stats.median(hohenheim_res_area2),2))+", sd="+str(round(stats.stdev(hohenheim_res_area2),2))+", N="+str(len(hohenheim_res_area2)))
axs[0].hist(hohenheim_res_area, bins, alpha=0.5, color="black")
axs[1].hist(hohenheim_res_area2, bins, alpha=0.5, color="black")
axs[0].set_ylabel('counts')
axs[1].set_xlabel('size (micrometers**2)')
axs[1].set_ylabel('counts')
fig.savefig("hohenheim_hist_annoV2_px025_px05034_visc_testMultiInferenceMergedPolys_v_nonMerged_23mar.png")
fig.clf()

#############################################


dirName = "/well/lindgren/users/swf744/git/dev-happy/projects/adipose/trainingWithMunichLeipzigHohenheimV2_testMultiInferenceMergedPolys_21mar/"
dirName2 = "/well/lindgren/users/swf744/git/dev-happy/projects/adipose/trainingWithMunichLeipzigHohenheimV2_testMultiInference_09mar/"
files = os.listdir(dirName)
hohenheimFiles =  [f for f in files if f.startswith('h') and "0.25" in f and "sc" in f and f.endswith('.geojson')]

hohenheim_list = []
hohenheim_list2 = []
for i in hohenheimFiles:
    print("i")
    print(i)
    res = readGeoJSON2list(dirName+i)
    hohenheim_list.append(res)
    res2 = readGeoJSON2list(dirName2+i.replace("_px1_0.25_px2_0.5034","_px0.25").replace(".svs_merged","coords_merged_segmodel1_overlap256"))
    hohenheim_list2.append(res2)


hohenheim_res = [item for sublist in hohenheim_list for item in sublist]
hohenheim_res2 = [item for sublist in hohenheim_list2 for item in sublist]

hohenheim_res_area = []
hohenheim_res_area2 = []

for poly in hohenheim_res:
    if poly.area*hohenheim_pixelSize**2 >= 200 and poly.area*hohenheim_pixelSize**2 <= 16000 and ((4*math.pi*poly.area ) / ((poly.length)**2)) > 0.75:
        hohenheim_res_area.append(poly.area*hohenheim_pixelSize**2)
for poly in hohenheim_res2:
    if poly.area*hohenheim_pixelSize**2 >= 200 and poly.area*hohenheim_pixelSize**2 <= 16000 and ((4*math.pi*poly.area ) / ((poly.length)**2)) > 0.75:
        hohenheim_res_area2.append(poly.area*hohenheim_pixelSize**2)

bins = np.linspace(200, 16000, 100)

fig, axs = pyplot.subplots(2)
fig.suptitle("merged-size: mean="+str(round(stats.mean(hohenheim_res_area),2))+", med="+str(round(stats.median(hohenheim_res_area),2))+", sd="+str(round(stats.stdev(hohenheim_res_area),2))+", N="+str(len(hohenheim_res_area))+"\n"
             "size: mean="+str(round(stats.mean(hohenheim_res_area2),2))+", med="+str(round(stats.median(hohenheim_res_area2),2))+", sd="+str(round(stats.stdev(hohenheim_res_area2),2))+", N="+str(len(hohenheim_res_area2)))
axs[0].hist(hohenheim_res_area, bins, alpha=0.5, color="black")
axs[1].hist(hohenheim_res_area2, bins, alpha=0.5, color="black")
axs[0].set_ylabel('counts')
axs[1].set_xlabel('size (micrometers**2)')
axs[1].set_ylabel('counts')
fig.savefig("hohenheim_hist_annoV2_px025_px05034_subc_testMultiInferenceMergedPolys_v_nonMerged_23mar.png")
fig.clf()

################################################################


dirName = "/well/lindgren/users/swf744/git/dev-happy/projects/adipose/trainingWithMunichLeipzigHohenheimV2_testMultiInferenceMergedPolys_21mar/"
dirName2 = "/well/lindgren/users/swf744/git/dev-happy/projects/adipose/trainingWithMunichLeipzigHohenheimV2_testMultiInference_09mar/"
files = os.listdir(dirName)
endoxFiles =  [f for f in files if f.startswith('Image') and "0.25" in f and f.endswith('.geojson')]

endox_list = []
endox_list2 = []
for i in endoxFiles:
    print("i")
    print(i)
    res = readGeoJSON2list(dirName+i)
    endox_list.append(res)
    res2 = readGeoJSON2list(dirName2+i.replace("_px1_0.25_px2_0.5034","_px0.25").replace(".scn_merged","coords_merged_segmodel1_overlap256"))
    endox_list2.append(res2)


endox_res = [item for sublist in endox_list for item in sublist]
endox_res2 = [item for sublist in endox_list2 for item in sublist]

endox_res_area = []
endox_res_area2 = []

for poly in endox_res:
    if poly.area*endox_pixelSize**2 >= 200 and poly.area*endox_pixelSize**2 <= 16000 and ((4*math.pi*poly.area ) / ((poly.length)**2)) > 0.75:
        endox_res_area.append(poly.area*endox_pixelSize**2)
for poly in endox_res2:
    if poly.area*endox_pixelSize**2 >= 200 and poly.area*endox_pixelSize**2 <= 16000 and ((4*math.pi*poly.area ) / ((poly.length)**2)) > 0.75:
        endox_res_area2.append(poly.area*endox_pixelSize**2)

bins = np.linspace(200, 16000, 100)

fig, axs = pyplot.subplots(2)
fig.suptitle("merged-size: mean="+str(round(stats.mean(endox_res_area),2))+", med="+str(round(stats.median(endox_res_area),2))+", sd="+str(round(stats.stdev(endox_res_area),2))+", N="+str(len(endox_res_area))+"\n"
             "size: mean="+str(round(stats.mean(endox_res_area2),2))+", med="+str(round(stats.median(endox_res_area2),2))+", sd="+str(round(stats.stdev(endox_res_area2),2))+", N="+str(len(endox_res_area2)))
axs[0].hist(endox_res_area, bins, alpha=0.5, color="black")
axs[1].hist(endox_res_area2, bins, alpha=0.5, color="black")
axs[0].set_ylabel('counts')
axs[1].set_xlabel('size (micrometers**2)')
axs[1].set_ylabel('counts')
fig.savefig("endox_hist_annoV2_px025_px05034_testMultiInferenceMergedPolys_v_nonMerged_23mar.png")
fig.clf()


########################################################


dirName = "/well/lindgren/users/swf744/git/dev-happy/projects/adipose/trainingWithMunichLeipzigHohenheimV2_testMultiInferenceMergedPolys_21mar/"
dirName2 = "/well/lindgren/users/swf744/git/dev-happy/projects/adipose/trainingWithMunichLeipzigHohenheimV2_testMultiInference_09mar/"
files = os.listdir(dirName)
gtexFiles =  [f for f in files if f.startswith('GTEX') and "0.25" in f and "Subcutaneous" in f and f.endswith('.geojson')]

gtex_list = []
gtex_list2 = []
for i in gtexFiles:
    print("i")
    print(i)
    res = readGeoJSON2list(dirName+i)
    gtex_list.append(res)
    res2 = readGeoJSON2list(dirName2+i.replace("_px1_0.25_px2_0.5034","_px0.25").replace(".svs_merged","coords_merged_segmodel1_overlap256"))
    gtex_list2.append(res2)


gtex_res = [item for sublist in gtex_list for item in sublist]
gtex_res2 = [item for sublist in gtex_list2 for item in sublist]

gtex_res_area = []
gtex_res_area2 = []

for poly in gtex_res:
    if poly.area*gtex_pixelSize**2 >= 200 and poly.area*gtex_pixelSize**2 <= 16000 and ((4*math.pi*poly.area ) / ((poly.length)**2)) > 0.75:
        gtex_res_area.append(poly.area*gtex_pixelSize**2)
for poly in gtex_res2:
    if poly.area*gtex_pixelSize**2 >= 200 and poly.area*gtex_pixelSize**2 <= 16000 and ((4*math.pi*poly.area ) / ((poly.length)**2)) > 0.75:
        gtex_res_area2.append(poly.area*gtex_pixelSize**2)

bins = np.linspace(200, 16000, 100)

fig, axs = pyplot.subplots(2)
fig.suptitle("merged-size: mean="+str(round(stats.mean(gtex_res_area),2))+", med="+str(round(stats.median(gtex_res_area),2))+", sd="+str(round(stats.stdev(gtex_res_area),2))+", N="+str(len(gtex_res_area))+"\n"
             "size: mean="+str(round(stats.mean(gtex_res_area2),2))+", med="+str(round(stats.median(gtex_res_area2),2))+", sd="+str(round(stats.stdev(gtex_res_area2),2))+", N="+str(len(gtex_res_area2)))
axs[0].hist(gtex_res_area, bins, alpha=0.5, color="black")
axs[1].hist(gtex_res_area2, bins, alpha=0.5, color="black")
axs[0].set_ylabel('counts')
axs[1].set_xlabel('size (micrometers**2)')
axs[1].set_ylabel('counts')
fig.savefig("gtex_hist_annoV2_px025_px05034_subc_testMultiInferenceMergedPolys_v_nonMerged_23mar.png")
fig.clf()



##############################################################

dirName = "/well/lindgren/users/swf744/git/dev-happy/projects/adipose/trainingWithMunichLeipzigHohenheimV2_testMultiInferenceMergedPolys_21mar/"
dirName2 = "/well/lindgren/users/swf744/git/dev-happy/projects/adipose/trainingWithMunichLeipzigHohenheimV2_testMultiInference_09mar/"
files = os.listdir(dirName)
gtexFiles =  [f for f in files if f.startswith('GTEX') and "0.25" in f and "Visceral" in f and f.endswith('.geojson')]

gtex_list = []
gtex_list2 = []
for i in gtexFiles:
    print("i")
    print(i)
    res = readGeoJSON2list(dirName+i)
    gtex_list.append(res)
    res2 = readGeoJSON2list(dirName2+i.replace("_px1_0.25_px2_0.5034","_px0.25").replace(".svs_merged","coords_merged_segmodel1_overlap256"))
    gtex_list2.append(res2)


gtex_res = [item for sublist in gtex_list for item in sublist]
gtex_res2 = [item for sublist in gtex_list2 for item in sublist]

gtex_res_area = []
gtex_res_area2 = []

for poly in gtex_res:
    if poly.area*gtex_pixelSize**2 >= 200 and poly.area*gtex_pixelSize**2 <= 16000 and ((4*math.pi*poly.area ) / ((poly.length)**2)) > 0.75:
        gtex_res_area.append(poly.area*gtex_pixelSize**2)
for poly in gtex_res2:
    if poly.area*gtex_pixelSize**2 >= 200 and poly.area*gtex_pixelSize**2 <= 16000 and ((4*math.pi*poly.area ) / ((poly.length)**2)) > 0.75:
        gtex_res_area2.append(poly.area*gtex_pixelSize**2)

bins = np.linspace(200, 16000, 100)

fig, axs = pyplot.subplots(2)
fig.suptitle("merged-size: mean="+str(round(stats.mean(gtex_res_area),2))+", med="+str(round(stats.median(gtex_res_area),2))+", sd="+str(round(stats.stdev(gtex_res_area),2))+", N="+str(len(gtex_res_area))+"\n"
             "size: mean="+str(round(stats.mean(gtex_res_area2),2))+", med="+str(round(stats.median(gtex_res_area2),2))+", sd="+str(round(stats.stdev(gtex_res_area2),2))+", N="+str(len(gtex_res_area2)))
axs[0].hist(gtex_res_area, bins, alpha=0.5, color="black")
axs[1].hist(gtex_res_area2, bins, alpha=0.5, color="black")
axs[0].set_ylabel('counts')
axs[1].set_xlabel('size (micrometers**2)')
axs[1].set_ylabel('counts')
fig.savefig("gtex_hist_annoV2_px025_px05034_visc_testMultiInferenceMergedPolys_v_nonMerged_23mar.png")
fig.clf()



