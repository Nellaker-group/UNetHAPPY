from shapely.geometry import Polygon, MultiPolygon, shape  # (pip install Shapely)

import db.eval_runs_interface as db
import data.geojsoner as gj
import data.merge_polygons as mp
import argparse
from peewee import fn
import sys

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

###################################################################

prs = argparse.ArgumentParser()
prs.add_argument('--db-name', help='Name of database to merge two EvalRuns whose polygons will be merged a .geojson file', type=str)
prs.add_argument('--run-id1', help='EvalRun_id of first run (int)', type=int)
prs.add_argument('--run-id2', help='EvalRun_id of second run (int)', type=int)
args = vars(prs.parse_args())

db_name = args['db_name']
run_id1 = args['run_id1']
run_id2 = args['run_id2']
assert db_name != ""

db.init(str(db_name))

seg_preds1 = db.get_all_validated_seg_preds(run_id1)
slide_name1 = db.get_slide_name(run_id1)

seg_preds2 = db.get_all_validated_seg_preds(run_id2)
slide_name2 = db.get_slide_name(run_id2)

run1 = db.get_eval_run_by_id(run_id1)
run2 = db.get_eval_run_by_id(run_id2)

assert slide_name1 == slide_name2
assert run1.pixel_size != run2.pixel_size

polys1 = db_to_list(seg_preds1)
polys2 = db_to_list(seg_preds2)

polys = [polys1, polys2]

polys_flat=[x for xs in polys for x in xs]

merged_polys_list = mp.merge_polysV3(polys_flat)

gj.writeToGeoJSON(merged_polys_list, slideName1+'_merged_px1'+str(run1.pixel_size)+'_px2'+str(run2.pixel_size)'.geojson')




