from typing import List

import typer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Polygon
import math
from matplotlib import pyplot

import sys
sys.path.insert(0, '/well/lindgren/users/swf744/git/HAPPY/projects/adipose')

#from happy.organs.organs import get_organ
#from happy.utils.utils import get_project_dir
import db.eval_runs_interface as db
from data.geojsoner import writeToGeoJSON

def main(
    run_id: int = 0,
    project_name: str = "adipose",
    #model_weights_dir: str = typer.Option(...),
    #model_name: str = typer.Option(...),
):
    # Create database connection
    db.init()

    eval_run = db.get_eval_run_by_id(run_id)
    print("eval_run")
    print(eval_run)
    
    seg_preds = db.get_all_validated_seg_preds(run_id)
    seg_list = []

    for seg in seg_preds:
        poly=Polygon([(x,y) for x,y in db.stringListTuple2coordinates(seg)])
        seg_list.append(poly)

    slide_name = db.get_slide_name(run_id)
    geojson_name = slide_name.replace(".scn",".geojson").replace(" ","_")

    writeToGeoJSON(seg_list, geojson_name)


    area_list = []
    for ele in seg_list:
        area_list.append(ele.area/16)

    pyplot.hist(area_list, 100)
    pyplot.savefig(geojson_name.replace(".geojson","_histSize.png"))
    pyplot.clf()

    area_list_PP = []
    for ele in seg_list:
        area_list_PP.append((4*math.pi*ele.area ) / ((ele.length)**2))

    pyplot.hist(area_list_PP, 100)
    pyplot.savefig(geojson_name.replace(".geojson","_histSize_PP.png"))
    pyplot.clf()
        

        
    area_list_filtered_areas  = []        
    area_list_filteredV2_areas = []
    area_list_filtered_polygons = []
    # glastonbury et al filtering of size above 200 and below 1600 micro square metre
    for ele in seg_list:
        if (ele.area / 16) > 200 and (ele.area / 16) < 16000:
            area_list_filtered_polygons.append(ele)
            area_list_filtered_areas.append(ele.area/16)
            if ((4*math.pi*ele.area) / ((ele.length)**2)) > 0.6:
                area_list_filteredV2_areas.append(ele.area/16)

    pyplot.hist(area_list_filtered_areas, 100)
    pyplot.savefig(geojson_name.replace(".geojson","_histSizeFilteredByArea.png"))
    pyplot.clf()

    pyplot.hist(area_list_filteredV2_areas, 100)
    pyplot.savefig(geojson_name.replace(".geojson","_histSizeFilteredByAreaAndPP.png"))
    pyplot.clf()


    area_list_filtered_PP = []
    for ele in area_list_filtered_polygons:
        area_list_filtered_PP.append((4*math.pi*ele.area ) / ((ele.length)**2))


    pyplot.hist(area_list_filtered_PP, 100)
    pyplot.savefig(geojson_name.replace(".geojson","_histSizeFilteredByArea_PP.png"))
    pyplot.clf()



if __name__ == "__main__":
    typer.run(main)
