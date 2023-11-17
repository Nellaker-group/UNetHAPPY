import os
print(os.environ["CONDA_PREFIX"])
print(os.environ['CONDA_DEFAULT_ENV'])

import sys
import os
# to be able to read in the libaries properly
sys.path.append(os.getcwd())

import random
from matplotlib import pyplot
import seaborn as sns
import pandas as pd
import statistics as stats
import math
import db.eval_runs_interface as db
import data.geojsoner as gj
import data.merge_polygons as mp
import argparse
from peewee import fn
from db.eval_runs import EvalRun, TileState, Prediction, UnvalidatedPrediction, MergedPrediction

import shapely
from shapely.geometry import Polygon, MultiPolygon, shape  # (pip install Shapely)

#################################################################

def db_to_list(seg_preds):
    seg_list = []
    poly_list = []
    old_poly_id = seg_preds[0][0]
    for coord in seg_preds:
        if old_poly_id == coord[0]:
            poly_list.append((coord[2],coord[3]))
        else:
            poly=Polygon(poly_list)
            # checking polygon is valid
            if poly.is_valid:
                seg_list.append(poly)
            poly_list = []
            # for the first point when poly_id changes
            poly_list.append((coord[2],coord[3]))
            old_poly_id = coord[0]
    return(seg_list)

#################################################################

craigFilter=False


pixels = {}

pixels["leipzig"] = 0.5034
pixels["munich"] = 0.5034
pixels["hohenheim"] = 0.5034
pixels["gtex"] = 0.4942
pixels["endox"] = 0.2500

def simulate_cells(n,data,mean=True):
    sub=[]
    for i in range(100):
        if mean == True:
            sub.append(stats.mean(random.sample(data,n)))
        else:
            sub.append(stats.stdev(random.sample(data,n)))
    return (sub)


random.seed(1337)

def boxplot_sampling_individual(j,i,title):
    db.init(str(j).replace(str(j),"task_dbs/"+str(j)+".db"))

    max_run_id = EvalRun.select(fn.MAX(EvalRun.id)).scalar()

    ## we only want one slide
    max_run_id = i+2

    tmp_list = []
    below750_list = []

    for i in range(i,max_run_id,2):

        seg_preds = db.get_all_merged_seg_preds(i,i+1)
        slide_name = db.get_slide_name(i)
        used_slide_name = ""
        print(slide_name)

        indi_id = slide_name.split("/")[-1]
        used_slide_name = indi_id


        print(indi_id)

        if len(seg_preds) == 0:
            continue

        polys=db_to_list(seg_preds)
                
        whichPixel = ""
        if indi_id.startswith("a"):
            whichPixel = "leipzig"
        elif indi_id.startswith("m"):
            whichPixel = "munich"
        elif indi_id.startswith("h"):
            whichPixel = "hohenheim"
        elif indi_id.startswith("GTEX"):
            whichPixel = "gtex"
        elif indi_id.startswith("Image"):
            whichPixel = "endox"

        for poly in polys:
            if craigFilter:
                good = poly.area*pixels[whichPixel]**2 >= 200 and poly.area*pixels[whichPixel]**2 <= 16000
            else:
                good = poly.area*pixels[whichPixel]**2 >= 200 and ((4*math.pi*poly.area ) / ((poly.length)**2)) > 0.75
            if good:
                tmp_list.append(poly.area*pixels[whichPixel]**2)
                if poly.area*pixels[whichPixel]**2 < 750:
                    below750_list.append(poly.area*pixels[whichPixel]**2)


    simulation = pd.DataFrame(
        {'1000': simulate_cells(1000, tmp_list),
         '500': simulate_cells(500, tmp_list),
         '100': simulate_cells(100, tmp_list),
         '10': simulate_cells(10, tmp_list),
         '3': simulate_cells(3, tmp_list)
     })


    pyplot.figure(figsize=(10,6))
    pyplot.rcParams["axes.labelsize"] = 20
    pyplot.rcParams["xtick.labelsize"] = 15
    pyplot.rcParams["ytick.labelsize"] = 15
    sns.set_style("white")
    ax = sns.boxplot(data=simulation,order=['3','10','100','500','1000'])
    ax = sns.swarmplot(data=simulation, color=".25",order=['3','10','100','500','1000'],alpha=.3)
    pyplot.xlabel('Number of cells sampled',fontsize=20)
    pyplot.ylabel('Average adipocyte area ($\mu m^{2}$)',fontsize=20)
    pyplot.title('Monte Carlo sampling of adipocytes, one individual - '+title,fontsize=20)

    pyplot.savefig("plotsPresentationCecilia/mc_sampling_"+indi_id+".png")
    pyplot.clf()




boxplot_sampling_individual(35,1,"Munich")
boxplot_sampling_individual(135,1,"Leipzig")
boxplot_sampling_individual(235,1,"GTEX")
boxplot_sampling_individual(7,15,"Hohenheim")
boxplot_sampling_individual(13,11,"Endox")







