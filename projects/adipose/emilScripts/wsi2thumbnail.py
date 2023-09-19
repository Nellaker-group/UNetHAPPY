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

import openslide as osl



prs = argparse.ArgumentParser()
prs.add_argument('--slideName', help='Path of slide', type=str)
args = vars(prs.parse_args())

assert args['slideName'] != ""

slideName = args['slideName']


slide = osl.OpenSlide(slideName)
thumb = slide.get_thumbnail((1024,1024))
thumb.save(os.path.basename(slideName.replace(".svs","_thumbnail.png").replace(".scn","_thumbnail.png")))





