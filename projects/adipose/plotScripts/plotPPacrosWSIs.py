
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
import re
import pickle
from peewee import fn
from db.eval_runs import EvalRun, TileState, Prediction, UnvalidatedPrediction, MergedPrediction

import shapely
from shapely.geometry import Polygon, MultiPolygon, shape  # (pip install Shapely)

from matplotlib import pyplot
import seaborn as sns

prs = argparse.ArgumentParser()
prs.add_argument('--i1', help='Start index (included)', type=int)
prs.add_argument('--i2', help='End index (included)', type=int)
args = vars(prs.parse_args())

assert args['i1'] != ""
assert args['i2'] != ""

i1 = args['i1']
i2 = args['i2']

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


def plotter(pp_leipzig,pp_munich,pp_hohenheim,pp_gtex,pp_endox,n_leipzig,n_munich,n_hohenheim,n_gtex,n_endox,depot,plotFile):
    # Create the figure and axes
    fig, axes = pyplot.subplots(3, 2, figsize=(16, 6), sharex=True)
    # Flatten the axes array for easier iteration
    #axes = axes.flatten()
    sns.histplot(pp_leipzig, ax=axes[0,0], kde=False, color="red", alpha=0.4, bins=50, stat="density")
    axes[0,0].set_title("Lepizig - "+depot+" N="+str(n_leipzig),size=12)
    axes[0,0].set_ylabel('density')
    axes[0,0].set_xlabel("Leipzig",size=12)
    axes[0,0].axvline(0.75, color='black')
    sns.histplot(pp_munich, ax=axes[0,1], kde=False, color="blue", alpha=0.4, bins=50, stat="density")
    axes[0,1].set_title("Munich - "+depot+" N="+str(n_munich),size=12)
    axes[0,1].set_ylabel('density')
    axes[0,1].set_xlabel("Munich",size=12)
    axes[0,1].axvline(0.75, color='black')
    sns.histplot(pp_hohenheim, ax=axes[1,0], kde=False, color="green", alpha=0.4, bins=50, stat="density")
    axes[1,0].set_title("Hohenheim - "+depot+" N="+str(n_hohenheim),size=12)
    axes[1,0].set_ylabel('density')
    axes[1,0].set_xlabel("Hohenheim",size=12)
    axes[1,0].axvline(0.75, color='black')
    sns.histplot(pp_gtex, ax=axes[1,1], kde=False, color="purple", alpha=0.4, bins=50, stat="density")
    axes[1,1].set_title("GTEX - "+depot+" N="+str(n_gtex),size=12)
    axes[1,1].set_ylabel('density')
    axes[1,1].set_xlabel("GTEX",size=12)
    axes[1,1].axvline(0.75, color='black')
    sns.histplot(pp_endox, ax=axes[2,1], kde=False, color="orange", alpha=0.4, bins=50, stat="density")
    axes[2,1].set_title("ENDOX - "+depot+" N="+str(n_endox),size=12)
    axes[2,1].set_ylabel('density')
    axes[2,1].set_xlabel("ENDOX",size=12)
    axes[2,1].axvline(0.75, color='black')
    # Adjust spacing between subplots
    pyplot.tight_layout()
    #pyplot.subplots_adjust(hspace=0.35)
    pyplot.savefig(plotFile)
    pyplot.clf()


def plotter2d(pp_leipzig,pp_munich,pp_hohenheim,pp_gtex,pp_endox,n_leipzig,n_munich,n_hohenheim,n_gtex,n_endox,size_leipzig,size_munich,size_hohenheim,size_gtex,size_endox,depot,plotFile):
    # Create the figure and axes
    #fig, axes = pyplot.subplots(2, 2, figsize=(16, 6), sharex=True)
    pyplot.figure(figsize=(12, 8))
    pyplot.subplot(3, 2, 1)
    pyplot.hist2d(pp_leipzig, size_leipzig, bins=25, density=False, cmap='viridis', range=[[0, 1], [0, 20000]])
    pyplot.title("Lepizig - "+depot+" N="+str(n_leipzig),size=12)
    pyplot.axvline(0.75, color='white')

    pyplot.subplot(3, 2, 2)
    pyplot.hist2d(pp_munich, size_munich, bins=25, density=False, cmap='viridis', range=[[0, 1], [0, 20000]])
    pyplot.title("Munich - "+depot+" N="+str(n_munich),size=12)
    pyplot.axvline(0.75, color='white')

    pyplot.subplot(3, 2, 3)
    pyplot.hist2d(pp_hohenheim, size_hohenheim,  bins=25, density=False, cmap='viridis', range=[[0, 1], [0, 20000]])
    pyplot.title("Hohenheim - "+depot+" N="+str(n_hohenheim),size=12)
    pyplot.axvline(0.75, color='white')
    
    pyplot.subplot(3, 2, 4)
    pyplot.hist2d(pp_gtex, size_gtex, bins=25, density=False, cmap='viridis', range=[[0, 1], [0, 20000]])
    pyplot.title("GTEX - "+depot+" N="+str(n_gtex),size=12)
    pyplot.axvline(0.75, color='white')

    pyplot.subplot(3, 2, 5)
    pyplot.hist2d(pp_endox, size_endox, bins=25, density=False, cmap='viridis', range=[[0, 1], [0, 20000]])
    pyplot.title("ENDOX - "+depot+" N="+str(n_endox),size=12)
    pyplot.axvline(0.75, color='white')


    # Adjust spacing between subplots
    pyplot.tight_layout()
    #pyplot.subplots_adjust(hspace=0.35)
    pyplot.savefig(plotFile)
    pyplot.clf()



def fractionEachBox(pp_list, size_list):
    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0
    for i in range(len(pp_list)):
        if pp_list[i] > 0.6 and size_list[i] > 2.5:
            count1 += 1
        elif pp_list[i] < 0.6 and size_list[i] > 2.5:
            count2 += 1
        elif pp_list[i] < 0.6 and size_list[i] < 2.5:
            count3 += 1
        else:
            count4 += 1
    if len(pp_list)==0:
        total_count = 1
    else:
        total_count = len(pp_list)        
    return((100*count1/total_count,100*count2/total_count,100*count3/total_count,100*count4/total_count))



def plotter2dlog(pp_leipzig,pp_munich,pp_hohenheim,pp_gtex,pp_endox,n_leipzig,n_munich,n_hohenheim,n_gtex,n_endox,size_leipzig,size_munich,size_hohenheim,size_gtex,size_endox,depot,plotFile):
    # Create the figure and axes
    #fig, axes = pyplot.subplots(2, 2, figsize=(16, 6), sharex=True)
    pyplot.figure(figsize=(20, 10))
    pyplot.subplot(3, 2, 1)
    log_size_leipzig = [math.log10(x) for x in size_leipzig]
    pyplot.hist2d(pp_leipzig, log_size_leipzig, bins=25, density=False, cmap='viridis')
    pyplot.title("Lepizig - "+depot+" N="+str(n_leipzig),size=12)
    pyplot.ylabel('log10(size)')
    #pyplot.axvline(0.75, color='white')
    #pyplot.axhline(2.30, color='white')
    pyplot.axvline(0.6, color='white')
    pyplot.axhline(2.50, color='white')
    pyplot.text(0.75, 5, "frac= "+str(round(fractionEachBox(pp_leipzig, log_size_leipzig)[0],2)), color='white', fontsize=12)
    pyplot.text(0.1, 5, "frac= "+str(round(fractionEachBox(pp_leipzig, log_size_leipzig)[1],2)), color='white', fontsize=12)
    pyplot.text(0.1, -0.5, "frac= "+str(round(fractionEachBox(pp_leipzig, log_size_leipzig)[2],2)), color='white', fontsize=12)
    pyplot.text(0.75, -0.5, "frac= "+str(round(fractionEachBox(pp_leipzig, log_size_leipzig)[3],2)), color='white', fontsize=12)
    pyplot.subplot(3, 2, 2)
    log_size_munich = [math.log10(x) for x in size_munich]
    pyplot.hist2d(pp_munich, log_size_munich, bins=25, density=False, cmap='viridis')
    pyplot.title("Munich - "+depot+" N="+str(n_munich),size=12)
    pyplot.ylabel('log10(size)')
    #pyplot.axvline(0.75, color='white')
    #pyplot.axhline(2.30, color='white')
    pyplot.axvline(0.6, color='white')
    pyplot.axhline(2.50, color='white')
    pyplot.text(0.75, 5, "frac= "+str(round(fractionEachBox(pp_munich, log_size_munich)[0],2)), color='white', fontsize=12)
    pyplot.text(0.1, 5, "frac= "+str(round(fractionEachBox(pp_munich, log_size_munich)[1],2)), color='white', fontsize=12)
    pyplot.text(0.1, -0.5, "frac= "+str(round(fractionEachBox(pp_munich, log_size_munich)[2],2)), color='white', fontsize=12)
    pyplot.text(0.75, -0.5, "frac= "+str(round(fractionEachBox(pp_munich, log_size_munich)[3],2)), color='white', fontsize=12)
    pyplot.subplot(3, 2, 3)
    log_size_hohenheim = [math.log10(x) for x in size_hohenheim]
    pyplot.hist2d(pp_hohenheim, log_size_hohenheim,  bins=25, density=False, cmap='viridis')
    pyplot.title("Hohenheim - "+depot+" N="+str(n_hohenheim),size=12)
    pyplot.ylabel('log10(size)')
    #pyplot.axvline(0.75, color='white')
    #pyplot.axhline(2.30, color='white')
    pyplot.axvline(0.6, color='white')
    pyplot.axhline(2.50, color='white')
    pyplot.text(0.75, 5, "frac= "+str(round(fractionEachBox(pp_hohenheim, log_size_hohenheim)[0],2)), color='white', fontsize=12)
    pyplot.text(0.1, 5, "frac= "+str(round(fractionEachBox(pp_hohenheim, log_size_hohenheim)[1],2)), color='white', fontsize=12)
    pyplot.text(0.1, -0.5, "frac= "+str(round(fractionEachBox(pp_hohenheim, log_size_hohenheim)[2],2)), color='white', fontsize=12)
    pyplot.text(0.75, -0.5, "frac= "+str(round(fractionEachBox(pp_hohenheim, log_size_hohenheim)[3],2)), color='white', fontsize=12)
    pyplot.subplot(3, 2, 4)
    log_size_gtex = [math.log10(x) for x in size_gtex]
    pyplot.hist2d(pp_gtex, log_size_gtex, bins=25, density=False, cmap='viridis')
    pyplot.title("GTEX - "+depot+" N="+str(n_gtex),size=12) 
    pyplot.ylabel('log10(size)')
    #pyplot.axvline(0.75, color='white')
    #pyplot.axhline(2.30, color='white')
    pyplot.axvline(0.6, color='white')
    pyplot.axhline(2.50, color='white')
    pyplot.text(0.75, 5, "frac= "+str(round(fractionEachBox(pp_gtex, log_size_gtex)[0],2)), color='white', fontsize=12)
    pyplot.text(0.1, 5, "frac= "+str(round(fractionEachBox(pp_gtex, log_size_gtex)[1],2)), color='white', fontsize=12)
    pyplot.text(0.1, -0.5, "frac= "+str(round(fractionEachBox(pp_gtex, log_size_gtex)[2],2)), color='white', fontsize=12)
    pyplot.text(0.75, -0.5, "frac= "+str(round(fractionEachBox(pp_gtex, log_size_gtex)[3],2)), color='white', fontsize=12)
    pyplot.subplot(3, 2, 5)
    log_size_endox = [math.log10(x) for x in size_endox]
    pyplot.hist2d(pp_endox, log_size_endox, bins=25, density=False, cmap='viridis')
    pyplot.title("ENDOX - "+depot+" N="+str(n_endox),size=12) 
    pyplot.ylabel('log10(size)')
    #pyplot.axvline(0.75, color='white')
    #pyplot.axhline(2.30, color='white')
    pyplot.axvline(0.6, color='white')
    pyplot.axhline(2.50, color='white')
    pyplot.text(0.75, 5, "frac= "+str(round(fractionEachBox(pp_endox, log_size_endox)[0],2)), color='white', fontsize=12)
    pyplot.text(0.1, 5, "frac= "+str(round(fractionEachBox(pp_endox, log_size_endox)[1],2)), color='white', fontsize=12)
    pyplot.text(0.1, -0.5, "frac= "+str(round(fractionEachBox(pp_endox, log_size_endox)[2],2)), color='white', fontsize=12)
    pyplot.text(0.75, -0.5, "frac= "+str(round(fractionEachBox(pp_endox, log_size_endox)[3],2)), color='white', fontsize=12)
    # Adjust spacing between subplots
    pyplot.tight_layout()
    #pyplot.subplots_adjust(hspace=0.35)
    pyplot.savefig(plotFile)
    pyplot.clf()

####################################################################

pixels = {}

pixels["leipzig"] = 0.5034
pixels["munich"] = 0.5034
pixels["hohenheim"] = 0.5034
pixels["gtex"] = 0.4942
pixels["endox"] = 0.2500

n_leipzig_sc = 0
n_leipzig_vc = 0
n_munich_sc = 0
n_munich_vc = 0
n_hohenheim_sc = 0
n_hohenheim_vc = 0
n_gtex_sc = 0
n_gtex_vc = 0
n_endox_sc = 0
n_endox_vc = 0


pp_leipzig_sc = []
pp_leipzig_vc = []
pp_munich_sc = []
pp_munich_vc = []
pp_hohenheim_sc = []
pp_hohenheim_vc = []
pp_gtex_sc = []
pp_gtex_vc = []
pp_endox_sc = []
pp_endox_vc = []


size_leipzig_sc = []
size_leipzig_vc = []
size_munich_sc = []
size_munich_vc = []
size_hohenheim_sc = []
size_hohenheim_vc = []
size_gtex_sc = []
size_gtex_vc = []
size_endox_sc = []
size_endox_vc = []


big_pp_leipzig_sc = []
big_pp_leipzig_vc = []
big_pp_munich_sc = []
big_pp_munich_vc = []
big_pp_hohenheim_sc = []
big_pp_hohenheim_vc = []
big_pp_gtex_sc = []
big_pp_gtex_vc = []
big_pp_endox_sc = []
big_pp_endox_vc = []


big_size_leipzig_sc = []
big_size_leipzig_vc = []
big_size_munich_sc = []
big_size_munich_vc = []
big_size_hohenheim_sc = []
big_size_hohenheim_vc = []
big_size_gtex_sc = []
big_size_gtex_vc = []
big_size_endox_sc = []
big_size_endox_vc = []


endoxID = pd.read_csv("/gpfs3/well/lindgren/craig/isbi-2012/NDOG_histology_IDs_and_quality.csv")
endoxID_sc = endoxID[ endoxID["depot"] == "subcutaneous"]


for j in range(i1,i2+1):
    db.init(str(j).replace(str(j),"task_dbs/"+str(j)+".db"))
    max_run_id = EvalRun.select(fn.MAX(EvalRun.id)).scalar()
    for i in range(1,max_run_id,2):
        seg_preds = db.get_all_merged_seg_preds(i,i+1)
        slide_name = db.get_slide_name(i)
        used_slide_name = ""
        print(slide_name)
        indi_id = slide_name.split("/")[-1]
        used_slide_name = indi_id
        print(indi_id)
        sub = False
        if re.search(r"\dsc\d", indi_id) or "Subcutaneous" in indi_id or any(indi_id == endoxID_sc["filename"]):
            indi_id2 = indi_id.split("sc")[0]
            sub = True
        elif re.search(r"\dvc\d", indi_id) or "Visceral" in indi_id:
            indi_id2 = indi_id.split("vc")[0]
        else:
            indi_id2 = indi_id
        if len(seg_preds) == 0:
            continue
        polys=db_to_list(seg_preds)
        whichPixel = ""
        if indi_id.startswith("a"):
            whichPixel = "leipzig"
            for poly in polys:
                pp = (4*math.pi*poly.area) / (poly.length**2)
                size = poly.area*pixels[whichPixel]**2 
                if sub:
                    pp_leipzig_sc.append(pp)
                    size_leipzig_sc.append(size)
                    if size > 200:
                        big_pp_leipzig_sc.append(pp)
                        big_size_leipzig_sc.append(size)
                else:
                    pp_leipzig_vc.append(pp)
                    size_leipzig_vc.append(size)
                    if size > 200:
                        big_pp_leipzig_vc.append(pp)
                        big_size_leipzig_vc.append(size)
            if sub: 
                n_leipzig_sc += 1
            else: 
                n_leipzig_vc += 1
        elif indi_id.startswith("m"):
            whichPixel = "munich"
            for poly in polys:
                pp = (4*math.pi*poly.area) / (poly.length**2)
                size = poly.area*pixels[whichPixel]**2 
                if sub:
                    pp_munich_sc.append(pp)
                    size_munich_sc.append(size)
                    if size > 200:
                         big_pp_munich_sc.append(pp)
                         big_size_munich_sc.append(size)
                else:
                    pp_munich_vc.append(pp)
                    size_munich_vc.append(size)
                    if size > 200:
                        big_pp_munich_vc.append(pp)
                        big_size_munich_vc.append(size)
            if sub: 
                n_munich_sc += 1
            else: 
                n_munich_vc += 1
        elif indi_id.startswith("h"):
            whichPixel = "hohenheim"
            for poly in polys:
                pp = (4*math.pi*poly.area) / (poly.length**2)
                size = poly.area*pixels[whichPixel]**2 
                if sub:
                    pp_hohenheim_sc.append(pp)
                    size_hohenheim_sc.append(size)
                    if size > 200:
                        big_pp_hohenheim_sc.append(pp)
                        big_size_hohenheim_sc.append(size)
                else:
                    pp_hohenheim_vc.append(pp)
                    size_hohenheim_vc.append(size)
                    if size > 200:
                        big_pp_hohenheim_vc.append(pp)
                        big_size_hohenheim_vc.append(size)
            if sub: 
                n_hohenheim_sc += 1
            else: 
                n_hohenheim_vc += 1
        elif indi_id.startswith("GTEX"):
            whichPixel = "gtex"
            for poly in polys:
                pp = (4*math.pi*poly.area) / (poly.length**2)
                size = poly.area*pixels[whichPixel]**2 
                if sub:
                    pp_gtex_sc.append(pp)
                    size_gtex_sc.append(size)
                    if size > 200:
                        big_pp_gtex_sc.append(pp)
                        big_size_gtex_sc.append(size)
                else:
                    pp_gtex_vc.append(pp)
                    size_gtex_vc.append(size)
                    if size > 200:
                        big_pp_gtex_vc.append(pp)
                        big_size_gtex_vc.append(size)
            if sub: 
                n_gtex_sc += 1
            else: 
                n_gtex_vc += 1
        elif indi_id.startswith("Image"):
            whichPixel = "endox"
            for poly in polys:
                pp = (4*math.pi*poly.area) / (poly.length**2)
                size = poly.area*pixels[whichPixel]**2 
                if sub:
                    pp_endox_sc.append(pp)
                    size_endox_sc.append(size)
                    if size > 200:
                        big_pp_endox_sc.append(pp)
                        big_size_endox_sc.append(size)
                else:
                    pp_endox_vc.append(pp)
                    size_endox_vc.append(size)
                    if size > 200:
                        big_pp_endox_vc.append(pp)
                        big_size_endox_vc.append(size)
            if sub: 
                n_endox_sc += 1
            else: 
                n_endox_vc += 1


            

# emil
print(pp_endox_sc[0:10])
print(size_endox_sc[0:10])

print(pp_endox_vc[0:10])
print(size_endox_vc[0:10])


dict_lists = {}

dict_lists["i1"] = i1
dict_lists["i2"] = i2


dict_lists["pp_leipzig_sc"] = pp_leipzig_sc
dict_lists["pp_munich_sc"] = pp_munich_sc
dict_lists["pp_hohenheim_sc"] = pp_hohenheim_sc 
dict_lists["pp_gtex_sc"] = pp_gtex_sc
dict_lists["pp_endox_sc"] = pp_endox_sc

dict_lists["n_leipzig_sc"] = n_leipzig_sc
dict_lists["n_munich_sc"] = n_munich_sc
dict_lists["n_hohenheim_sc"] = n_hohenheim_sc
dict_lists["n_gtex_sc"] = n_gtex_sc
dict_lists["n_endox_sc"] = n_endox_sc

dict_lists["size_leipzig_sc"] = size_leipzig_sc
dict_lists["size_munich_sc"] = size_munich_sc
dict_lists["size_hohenheim_sc"] = size_hohenheim_sc
dict_lists["size_gtex_sc"] = size_gtex_sc
dict_lists["size_endox_sc"] = size_endox_sc


dict_lists["pp_leipzig_vc"] = pp_leipzig_vc
dict_lists["pp_munich_vc"] = pp_munich_vc
dict_lists["pp_hohenheim_vc"] = pp_hohenheim_vc 
dict_lists["pp_gtex_vc"] = pp_gtex_vc
dict_lists["pp_endox_vc"] = pp_endox_vc

dict_lists["n_leipzig_vc"] = n_leipzig_vc
dict_lists["n_munich_vc"] = n_munich_vc
dict_lists["n_hohenheim_vc"] = n_hohenheim_vc
dict_lists["n_gtex_vc"] = n_gtex_vc
dict_lists["n_endox_vc"] = n_endox_vc

dict_lists["size_leipzig_vc"] = size_leipzig_vc
dict_lists["size_munich_vc"] = size_munich_vc
dict_lists["size_hohenheim_vc"] = size_hohenheim_vc
dict_lists["size_gtex_vc"] = size_gtex_vc
dict_lists["size_endox_vc"] = size_endox_vc


fileObj = open("plotPPacrosWSIs_from"+str(i1)+"_"+str(i2)+".obj", "wb")
pickle.dump(dict_lists,fileObj)
fileObj.close()


plotter(pp_leipzig_sc,pp_munich_sc,pp_hohenheim_sc,pp_gtex_sc,pp_endox_sc,n_leipzig_sc,n_munich_sc,n_hohenheim_sc,n_gtex_sc,n_endox_sc,"SC","plots/samplePPhist_sc_from"+str(i1)+"_"+str(i2)+".png")
plotter(pp_leipzig_vc,pp_munich_vc,pp_hohenheim_vc,pp_gtex_vc,pp_endox_vc,n_leipzig_vc,n_munich_vc,n_hohenheim_vc,n_gtex_vc,n_endox_vc,"VC","plots/samplePPhist_vc_from"+str(i1)+"_"+str(i2)+".png")

plotter2d(big_pp_leipzig_sc,big_pp_munich_sc,big_pp_hohenheim_sc,big_pp_gtex_sc,big_pp_endox_sc,n_leipzig_sc,n_munich_sc,n_hohenheim_sc,n_gtex_sc,n_endox_sc,big_size_leipzig_sc,big_size_munich_sc,big_size_hohenheim_sc,big_size_gtex_sc,big_size_endox_sc,"SC","plots/samplePPhist2d_sizeFilter_sc_from"+str(i1)+"_"+str(i2)+".png")
plotter2d(big_pp_leipzig_vc,big_pp_munich_vc,big_pp_hohenheim_vc,big_pp_gtex_vc,big_pp_endox_vc,n_leipzig_vc,n_munich_vc,n_hohenheim_vc,n_gtex_vc,n_endox_vc,big_size_leipzig_vc,big_size_munich_vc,big_size_hohenheim_vc,big_size_gtex_vc,big_size_endox_vc,"VC","plots/samplePPhist2d_sizeFilter_vc_from"+str(i1)+"_"+str(i2)+".png")


plotter2dlog(pp_leipzig_sc,pp_munich_sc,pp_hohenheim_sc,pp_gtex_sc,pp_endox_sc,n_leipzig_sc,n_munich_sc,n_hohenheim_sc,n_gtex_sc,n_endox_sc,size_leipzig_sc,size_munich_sc,size_hohenheim_sc,size_gtex_sc,size_endox_sc,"SC","plots/samplePPhist2d_log10_sc_from"+str(i1)+"_"+str(i2)+".png")
plotter2dlog(pp_leipzig_vc,pp_munich_vc,pp_hohenheim_vc,pp_gtex_vc,pp_endox_vc,n_leipzig_vc,n_munich_vc,n_hohenheim_vc,n_gtex_vc,n_endox_vc,size_leipzig_vc,size_munich_vc,size_hohenheim_vc,size_gtex_vc,size_endox_vc,"VC","plots/samplePPhist2d_log10_vc_from"+str(i1)+"_"+str(i2)+".png")

