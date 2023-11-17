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

pixelSize = 0.4942

poly_preds_tmp = []
dirName = "/gpfs3/well/lindgren/users/swf744/git/dev-happy/projects/adipose/runManySlides_6dec_overlap128_pickles_gtex/"
pickleFiles = os.listdir(dirName)
pickleFiles =  [f for f in pickleFiles if f.endswith('.obj')]

print("pickleFiles")
print(pickleFiles)
for i in pickleFiles:
    print("i")
    print(i)
    file = open(dirName+i,'rb')
    poly_tmp = pickle.load(file)
    file.close()
    poly_preds_tmp.append(poly_tmp)
    
poly_preds128 = [item for sublist in poly_preds_tmp for item in sublist]



poly_preds_tmp = []
dirName = "/gpfs3/well/lindgren/users/swf744/git/dev-happy/projects/adipose/runManySlides_8dec_overlap256_pickles_gtex/"
pickleFiles = os.listdir(dirName)
pickleFiles =  [f for f in pickleFiles if f.endswith('.obj')]

print("pickleFiles")
print(pickleFiles)
for i in pickleFiles:
    print("i")
    print(i)
    file = open(dirName+i,'rb')
    poly_tmp = pickle.load(file)
    file.close()
    poly_preds_tmp.append(poly_tmp)

poly_preds256 = [item for sublist in poly_preds_tmp for item in sublist]


poly_preds_tmp = []
dirName = "/gpfs3/well/lindgren/users/swf744/git/dev-happy/projects/adipose/runManySlides_7dec_overlap512_pickles_gtex/"
pickleFiles = os.listdir(dirName)
pickleFiles =  [f for f in pickleFiles if f.endswith('.obj')]

print("pickleFiles")
print(pickleFiles)
for i in pickleFiles:
    print("i")
    print(i)
    file = open(dirName+i,'rb')
    poly_tmp = pickle.load(file)
    file.close()
    poly_preds_tmp.append(poly_tmp)

poly_preds512 = [item for sublist in poly_preds_tmp for item in sublist]


poly_preds128_filter = []
poly_preds128_area = []
poly_preds128_pp = []

for poly in poly_preds128:
    if poly.area*pixelSize**2 >= 200 and poly.area*pixelSize**2 <= 16000 and ((4*math.pi*poly.area ) / ((poly.length)**2)) > 0.75:
        poly_preds128_filter.append(poly)
        poly_preds128_area.append(poly.area*pixelSize**2)
        poly_preds128_pp.append(((4*math.pi*poly.area ) / ((poly.length)**2)))
        

poly_preds256_filter = []
poly_preds256_area = []
poly_preds256_pp = []

for poly in poly_preds256:
    if poly.area*pixelSize**2 >= 200 and poly.area*pixelSize**2 <= 16000 and ((4*math.pi*poly.area ) / ((poly.length)**2)) > 0.75:
        poly_preds256_filter.append(poly)
        poly_preds256_area.append(poly.area*pixelSize**2)
        poly_preds256_pp.append(((4*math.pi*poly.area ) / ((poly.length)**2)))

  
poly_preds512_filter = []
poly_preds512_area = []
poly_preds512_pp = []

for poly in poly_preds512:
    if poly.area*pixelSize**2 >= 200 and poly.area*pixelSize**2 <= 16000 and ((4*math.pi*poly.area ) / ((poly.length)**2)) > 0.75:
        poly_preds512_filter.append(poly)
        poly_preds512_area.append(poly.area*pixelSize**2)
        poly_preds512_pp.append(((4*math.pi*poly.area ) / ((poly.length)**2)))


bins = np.linspace(200, 16000, 100)
pyplot.hist(poly_preds128_area, bins, alpha=0.5, color="black")
pyplot.title("size: mean="+str(round(stats.mean(poly_preds128_area),2))+", med="+str(round(stats.median(poly_preds128_area),2))+", sd="+str(round(stats.stdev(poly_preds128_area),2))+", N="+str(len(poly_preds128_area)))
pyplot.xlabel('size (micrometers**2)')
pyplot.ylabel('counts')
pyplot.savefig("runManySlides_6dec_overlap128_pickles_gtex_histV2.png")
pyplot.clf()



bins = np.linspace(200, 16000, 100)
pyplot.hist(poly_preds256_area, bins, alpha=0.5, color="blue")
pyplot.title("size: mean="+str(round(stats.mean(poly_preds256_area),2))+", med="+str(round(stats.median(poly_preds256_area),2))+", sd="+str(round(stats.stdev(poly_preds256_area),2))+", N="+str(len(poly_preds256_area)))
pyplot.xlabel('size (micrometers**2)')
pyplot.ylabel('counts')
pyplot.savefig("runManySlides_8dec_overlap256_pickles_gtex_histV2.png")
pyplot.clf()


bins = np.linspace(200, 16000, 100)
pyplot.hist(poly_preds512_area, bins, alpha=0.5, color="green")
pyplot.title("size: mean="+str(round(stats.mean(poly_preds512_area),2))+", med="+str(round(stats.median(poly_preds512_area),2))+", sd="+str(round(stats.stdev(poly_preds512_area),2))+", N="+str(len(poly_preds512_area)))
pyplot.xlabel('size (micrometers**2)')
pyplot.ylabel('counts')
pyplot.savefig("runManySlides_7dec_overlap512_pickles_gtex_histV2.png")
pyplot.clf()


bins = np.linspace(200, 16000, 100)
pyplot.hist(poly_preds128_area, bins, alpha=0.25, color="black")
pyplot.hist(poly_preds256_area, bins, alpha=0.25, color="blue")
pyplot.hist(poly_preds512_area, bins, alpha=0.25, color="green")
pyplot.title("black - 128 overlap, blue - 256 overlap, green - 512 overlap")
pyplot.xlabel('size (micrometers**2)')
pyplot.ylabel('counts')
pyplot.savefig("runManySlides_6_7_8dec_overlap128_256_512_pickles_gtex_histV2.png")

