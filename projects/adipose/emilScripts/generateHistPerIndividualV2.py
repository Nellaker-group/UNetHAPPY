import os
print(os.environ["CONDA_PREFIX"])
print(os.environ['CONDA_DEFAULT_ENV'])

import sys
import os
# to be able to read in the libaries properly
sys.path.append(os.getcwd())

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

from matplotlib import pyplot
import seaborn as sns

prs = argparse.ArgumentParser()
prs.add_argument('--i1', help='Start index (included)', type=int)
prs.add_argument('--i2', help='End index (included)', type=int)
prs.add_argument('--craigFilter', help='Whether to apply filtering like Craig', type=bool, default=False)
args = vars(prs.parse_args())

assert args['i1'] != ""
assert args['i2'] != ""

i1 = args['i1']
i2 = args['i2']
craigFilter = args['craigFilter']


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

def plotter(dict0, plotName,plotTitle):
    colours = {}
    colours["leipzig"]="green"
    colours["munich"]="red"
    colours["hohenheim"]="blue"
    colours["gtex"]="brown"
    colours["endox"]="black"

    for key in dict0.keys():
        whichPixel = ""
        if key.startswith("a"):
            whichPixel = "leipzig"
        elif key.startswith("m"):
            whichPixel = "munich"
        elif key.startswith("h"):
            whichPixel = "hohenheim"
        elif key.startswith("GTEX"):
            whichPixel = "gtex"
        elif key.startswith("Image"):
            whichPixel = "endox"
        sns.histplot(data=dict0[key], kde=True, element="poly", color=colours[whichPixel], stat="probability",fill=False)

    pyplot.plot([], [], label='leipzig', color='green',alpha=0.5)
    pyplot.plot([], [], label='munich', color='red')
    pyplot.plot([], [], label='hohenheim', color='blue')
    pyplot.plot([], [], label='gtex', color='brown')
    pyplot.plot([], [], label='endox', color='black')

    plt.xlim(0, 15000)

    pyplot.xlabel('Size of adipocytes (micro m**2)')
    pyplot.ylabel('Frequency')
    pyplot.title(plotTitle+" N="+str(len(dict0.keys())))
    pyplot.legend()



    pyplot.savefig(plotName)
    pyplot.clf()

####################################################################

pixels = {}

pixels["leipzig"] = 0.5034
pixels["munich"] = 0.5034
pixels["hohenheim"] = 0.5034
pixels["gtex"] = 0.4942
pixels["endox"] = 0.2500

leipzig = []

merged_dict = {}

sc_dict = {}
vc_dict = {}

file1 = open('polygonsFiltered_sc_from'+str(i1)+'_to'+str(i2)+'V2.txt', 'w')
file2 = open('polygonsFiltered_vc_from'+str(i1)+'_to'+str(i2)+'V2.txt', 'w')


for j in range(i1,i2+1):

    db.init(str(j).replace(str(j),"task_dbs/"+str(j)+".db"))

    max_run_id = EvalRun.select(fn.MAX(EvalRun.id)).scalar()

    for i in range(1,max_run_id,2):

        sc_list = []
        vc_list = []
        below750_list = []

        seg_preds = db.get_all_merged_seg_preds(i,i+1)
        slide_name = db.get_slide_name(i)
        used_slide_name = ""
        print(slide_name)

        indi_id = slide_name.split("/")[-1]
        used_slide_name = indi_id

        print(indi_id)        

        sub = False

        if "sc" in indi_id or "Subcutaneous" in indi_id:
            indi_id2 = indi_id.split("sc")[0]
            sub = True
        elif "vc" in indi_id or "Visceral" in indi_id:
            indi_id2 = indi_id.split("vc")[0]
        else:
            indi_id2 = indi_id

        if indi_id2 in merged_dict:
            continue
        else:
            merged_dict[indi_id2] = "1"

        print(indi_id2)


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
                if sub:
                    sc_list.append(poly.area*pixels[whichPixel]**2)
                else:
                    vc_list.append(poly.area*pixels[whichPixel]**2)
            else:
                if poly.area*pixels[whichPixel]**2 < 750:
                    below750_list.append(poly.area*pixels[whichPixel]**2)

        if len(sc_list) > 0:
            # Convert the numbers in the list to strings and join them with commas
            line = ",".join(str(num) for num in sc_list)
            # Write the line to the file
            file1.write(whichPixel+","+slide_name+","+str(len(sc_list))+","+line + "\n")

        if len(vc_list) > 0:
            # Convert the numbers in the list to strings and join them with commas
            line = ",".join(str(num) for num in vc_list)
            # Write the line to the file
            file2.write(whichPixel+","+slide_name+","+str(len(vc_list))+","+line + "\n")
        

file1.close()
file2.close()








