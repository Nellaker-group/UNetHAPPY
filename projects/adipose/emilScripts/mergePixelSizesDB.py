from shapely.geometry import Polygon, MultiPolygon, shape  # (pip install Shapely)

import db.eval_runs_interface as db
import data.geojsoner as gj
import data.merge_polygons as mp
import argparse
from peewee import fn
import sys


prs = argparse.ArgumentParser()
prs.add_argument('--database-id', help='EvalRun ID run whose polygons to be converted to a .geojson file', type=str)
args = vars(prs.parse_args())

database_id = args['database_id']
assert database_id != ""

db.init(str(database_id))

max_run_id = EvalRun.select(fn.MAX(EvalRun.id)).scalar()

# store polygons by slide - name
seg_dict = {}
merged_dict = {}

for i in range(max_run_id):

    seg_preds = db.get_all_validated_seg_preds(i)
    slide_name = db.get_slide_name(i)
    used_slide_name = ""

    if slide_name in seg_dict and slide_name+"_2" in seg_dict:
        print("Slide "+slide_name+" already exists twice in the database this script does not support the same slide_name in 3 copies")
        sys.exit()        
    elif slide_name in seg_dict:
         seg_dict[slide_name+"_2"] = []
         used_slide_name = slide_name+"_2"
    else:        
        seg_dict[slide_name] = []
        used_slide_name = slide_name

    old_poly_id = seg_preds[0][0]

    for coord in seg_preds:            
        if old_poly_id == coord[0]:
            poly_list.append((coord[2],coord[3]))
        else:
            poly=Polygon(poly_list)
            # checking polygon is valid
            if poly.is_valid:
                seg_dict[used_slide_name].append(poly)
            poly_list = []
            # for the first point when poly_id changes
            poly_list.append((coord[2],coord[3]))
            old_poly_id = coord[0]

    if slide_name in seg_dict and slide_name+"_2" in seg_dict:
        total_list = [seg_dict[used_slide_name], seg_dict[used_slide_name+"_2"]]
        flat_total_list = [x for xs in total_list for x in xs]
        merged_polys_list = mp.merge_polysV3(flat_total_list)
        merged_dict[slide_name] = merged_polys_list
