from shapely.geometry import Polygon
import pickle
import db.eval_runs_interface as db
from data.geojsoner import  writeToGeoJSON
import argparse
import math
from matplotlib import pyplot
import os
import statistics as stats


prs = argparse.ArgumentParser()
prs.add_argument('--pickleFile', help='pickle file to convert into a geojson file - should end with .obj', type=str,default=None)
prs.add_argument('--pickleDir', help='directory of pickle files to convert into plots', type=str,default=None)
prs.add_argument('--pixelSize', help='size of pixels for calculating the right area in square micrometers', type=str)
args = vars(prs.parse_args())
assert args['pickleFile'] != None or args['pickleDir'] != None


pickleFile = args['pickleFile']
pickleDir = args['pickleDir']
pixelSize = float(args['pixelSize'])

print("options:")
print(pickleFile)
print(pickleDir)
print(pixelSize)

poly_preds = None
poly_list = []
area_list = []
log_area_list = []
pp_list = []

saveName = ""

if(pickleDir==None):

    file = open(pickleFile,'rb')
    poly_preds = pickle.load(file)
    file.close()
    saveName = pickleFile

else:
    
    poly_preds_tmp = []
    fullPath = os.path.abspath(pickleDir)
    pickleFiles = os.listdir(fullPath+"/")
    print("pickleFiles")
    print(pickleFiles)
    for i in pickleFiles:
        print("i")
        print(i)
        file = open(fullPath+"/"+i,'rb')
        poly_tmp = pickle.load(file)
        file.close()
        poly_preds_tmp.append(poly_tmp)
    poly_preds = [item for sublist in poly_preds_tmp for item in sublist]
    saveName = pickleDir + pickleDir.replace("/","") + ".obj"


for poly in poly_preds:
    poly_list.append(poly)
    area_list.append(poly.area*pixelSize**2)
    log_area_list.append(math.log(poly.area*pixelSize**2,10))
    pp_list.append(((4*math.pi*poly.area ) / ((poly.length)**2)))



pyplot.hist(area_list, 100)
pyplot.title("size: mean="+str(round(stats.mean(area_list),2))+", med="+str(round(stats.median(area_list),2))+", sd="+str(round(stats.stdev(area_list),2))+", N="+str(len(area_list)))
pyplot.xlabel('size (micrometers**2)')
pyplot.ylabel('counts')
pyplot.savefig(saveName.replace(".obj","_histSize.png"))
pyplot.clf()



pyplot.hist(pp_list, 10)
pyplot.xlabel('PP')
pyplot.ylabel('counts')
pyplot.savefig(saveName.replace(".obj","_histPP.png"))
pyplot.clf()



pyplot.scatter(pp_list, area_list)
pyplot.xlabel('PP')
pyplot.ylabel('size (micrometers**2)')
pyplot.title('size v. PP')
pyplot.savefig(saveName.replace(".obj","_scatterSizePP.png"))
pyplot.clf()

pyplot.scatter(pp_list, log_area_list)
pyplot.xlabel('PP')
pyplot.ylabel('size (micrometers**2)')
pyplot.title('size v. PP')
pyplot.savefig(saveName.replace(".obj","_scatterSizePPlog.png"))
pyplot.clf()


pyplot.hist2d(pp_list, area_list,bins=[50,50])
pyplot.xlabel('PP')
pyplot.ylabel('size (micrometers**2)')

pyplot.savefig(saveName.replace(".obj","_2Dhist.png"))

pyplot.clf()


pyplot.hist2d(pp_list, log_area_list,bins=[50,50])
pyplot.xlabel('PP')
pyplot.ylabel('log10(size (micrometers**2))')
pyplot.savefig(saveName.replace(".obj","_log2Dhist.png"))
pyplot.clf()


poly_list2 = []
area_list2 = []
pp_list2 = []


poly_list2_out = []
area_list2_out = []
pp_list2_out = []


for poly in poly_preds:
    if poly.area*pixelSize**2 >= 200 and poly.area*pixelSize**2 <= 16000:
        poly_list2.append(poly)
        area_list2.append(poly.area*pixelSize**2)
        pp_list2.append(((4*math.pi*poly.area ) / ((poly.length)**2)))
    else:
        poly_list2_out.append(poly)
        area_list2_out.append(poly.area*pixelSize**2)
        pp_list2_out.append(((4*math.pi*poly.area ) / ((poly.length)**2)))



pyplot.hist(area_list2, 100)
pyplot.title("size: mean="+str(round(stats.mean(area_list2),2))+", med="+str(round(stats.median(area_list2),2))+", sd="+str(round(stats.stdev(area_list2),2))+", N="+str(len(area_list2)))
pyplot.xlabel('size (micrometers**2)')
pyplot.ylabel('counts')
pyplot.savefig(saveName.replace(".obj","_histSizeFilteredOnSize.png"))
pyplot.clf()


pyplot.hist(pp_list2, 10)
pyplot.xlabel('PP')
pyplot.ylabel('counts')
pyplot.savefig(saveName.replace(".obj","_histPP_FilteredOnSize.png"))
pyplot.clf()

writeToGeoJSON(poly_list2, saveName.replace(".obj","_FilteredOnSize.geojson"))



pyplot.hist(area_list2_out, 100)
pyplot.title("size: mean="+str(round(stats.mean(area_list2_out),2))+", med="+str(round(stats.median(area_list2_out),2))+", sd="+str(round(stats.stdev(area_list2_out),2))+", N="+str(len(area_list2_out)))
pyplot.xlabel('size (micrometers**2)')
pyplot.ylabel('counts')
pyplot.savefig(saveName.replace(".obj","_histSizeFilteredOnSize_Out.png"))
pyplot.clf()


pyplot.hist(pp_list2_out, 10)
pyplot.xlabel('PP')
pyplot.ylabel('counts')
pyplot.savefig(saveName.replace(".obj","_histPP_FilteredOnSize_Out.png"))
pyplot.clf()

writeToGeoJSON(poly_list2_out, saveName.replace(".obj","_FilteredOnSize_Out.geojson"))



poly_list3 = []
area_list3 = []
pp_list3 = []


poly_list3_out = []
area_list3_out = []
pp_list3_out = []



for poly in poly_preds:
    if poly.area*pixelSize**2 >= 200 and poly.area*pixelSize**2 <= 16000 and ((4*math.pi*poly.area ) / ((poly.length)**2)) > 0.75:
        poly_list3.append(poly)
        area_list3.append(poly.area*pixelSize**2)
        pp_list3.append(((4*math.pi*poly.area ) / ((poly.length)**2)))
    else:
        poly_list3_out.append(poly)
        area_list3_out.append(poly.area*pixelSize**2)
        pp_list3_out.append(((4*math.pi*poly.area ) / ((poly.length)**2)))




pyplot.hist(area_list3, 100)
pyplot.title("size: mean="+str(round(stats.mean(area_list3),2))+", med="+str(round(stats.median(area_list3),2))+", sd="+str(round(stats.stdev(area_list3),2))+", N="+str(len(area_list3)))
pyplot.xlabel('size (micrometers**2)')
pyplot.ylabel('counts')
pyplot.savefig(saveName.replace(".obj","_histSizeFilteredOnSizeAndPP.png"))
pyplot.clf()


pyplot.hist(pp_list3, 10)
pyplot.xlabel('PP')
pyplot.ylabel('counts')
pyplot.savefig(saveName.replace(".obj","_histPP_FilteredOnSizeAndPP.png"))
pyplot.clf()

writeToGeoJSON(poly_list3, saveName.replace(".obj","_FilteredOnSizeAndPP.geojson"))

pyplot.hist(area_list3_out, 100)

pyplot.title("size: mean="+str(round(stats.mean(area_list3_out),2))+", med="+str(round(stats.median(area_list3_out),2))+", sd="+str(round(stats.stdev(area_list3_out),2))+", N="+str(len(area_list3_out)))
pyplot.xlabel('size (micrometers**2)')
pyplot.ylabel('counts')
pyplot.savefig(saveName.replace(".obj","_histSizeFilteredOnSizeAndPP_Out.png"))
pyplot.clf()


pyplot.hist(pp_list3_out, 10)
pyplot.xlabel('size (micrometers**2)')
pyplot.ylabel('counts')
pyplot.savefig(saveName.replace(".obj","_histPP_FilteredOnSizeAndPP_Out.png"))
pyplot.clf()

writeToGeoJSON(poly_list3_out, saveName.replace(".obj","_FilteredOnSizeAndPP_Out.geojson"))
