import sys
import os
# to be able to read in the libaries properly
sys.path.append(os.getcwd())

import numpy as np
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

csv = pd.read_csv("/gpfs3/well/lindgren/users/swf744/adipocyte/data/raw/mergedPhenotypeFile.csv")

columns = list(range(0,12)) + [13]
csv1 = csv[csv.columns[columns]]

csv1 = csv1.rename(columns={'avg_vc': 'Mu Area'})


csv1["Depot"] = "Visceral"


columns = list(range(0,12)) + [12]
csv2 = csv[csv.columns[columns]]

csv2 = csv2.rename(columns={'avg_sc': 'Mu Area'})


csv2["Depot"] = "Subcutaneous"

csvTotal = pd.concat([csv1, csv2])
csvTotal.Sex.replace([1.0, 2.0,151.0,160.0],['XY', 'XX',np.nan,np.nan], inplace=True)

#subq_pheno_merged.GENDER.replace(['M', 'F'],['XY', 'XX'], inplace=True)
#subq_pheno_merged.depot.replace([0, 1],['subcutaneous', 'visceral'], inplace=True)
fig = pyplot.figure(figsize=(15,10))
ax=sns.violinplot(y='Mu Area',x='Depot',hue='Sex',data=csvTotal)
ax.set(xlabel="Adipose Depot", ylabel="Mean Adipocyte Area")



pyplot.savefig("plotsPresentationCecilia/testSlide31V2.png")
