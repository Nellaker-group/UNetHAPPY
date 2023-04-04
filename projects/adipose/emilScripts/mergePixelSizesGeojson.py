from shapely.geometry import Polygon, MultiPolygon, shape  # (pip install Shapely)

import os
import sys
# to be able to read in the libaries properly
sys.path.append(os.getcwd())


import db.eval_runs_interface as db
import data.geojsoner as gj
import data.merge_polygons as mp
import argparse
from peewee import fn


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
prs.add_argument('--write-geojson', help='Writes a .geojson file of the merged polygons', type=bool, default=False)
prs.add_argument('--write-little-file', help='Writes an empty file indicating this run has finished', type=bool, default=False)
args = vars(prs.parse_args())

db_name = args['db_name']
run_id1 = args['run_id1']
run_id2 = args['run_id2']
write_geojson = args['write_geojson']
write_little_file = args['write_little_file']
assert db_name != ""

db.init(str(db_name))

seg_preds1 = db.get_all_validated_seg_preds(run_id1)
slide_name1 = db.get_slide_name(run_id1)

seg_preds2 = db.get_all_validated_seg_preds(run_id2)
slide_name2 = db.get_slide_name(run_id2)

run1 = db.get_eval_run_by_id(run_id1)
run2 = db.get_eval_run_by_id(run_id2)

print(slide_name1)
print(slide_name2)


assert slide_name1 == slide_name2
assert run1.pixel_size != run2.pixel_size

polys1 = db_to_list(seg_preds1)
polys2 = db_to_list(seg_preds2)

polys = [polys1, polys2]

polys_flat = [x for xs in polys for x in xs]

merged_polys_list = mp.merge_polysV3(polys_flat)


merged_coords = []
poly_id=0        
for poly in merged_polys_list:            
    point_id = 0
    # it updates with a new poly_id for the merged polygons
    if poly.type == 'Polygon':
        for x,y in poly.exterior.coords:
            merged_coords.append((poly_id,point_id,int(x),int(y)))
            point_id += 1
        if poly.type == 'MultiPolygon':
            coordslist = [x.exterior.coords for x in poly.geoms]
            tmpcoordslist=[x for xs in coordslist for x in xs]
            for x,y in tmpcoordslist:
                merged_coords.append((poly_id,point_id,int(x),int(y)))
                point_id += 1
        poly_id += 1        
        # push the predictions to the database

db.validate_merged_workings(run_id1, run_id2, merged_coords)

if write_geojson:
    gj.writeToGeoJSON(merged_polys_list, os.getcwd()+"/"+slide_name1.split("/")[-1]+'_merged_px1_'+str(run1.pixel_size)+'_px2_'+str(run2.pixel_size)+'.geojson')

if write_little_file:
    f = open(os.getcwd()+"/littleLogFiles/"+slide_name1.split("/")[-1]+"_id1_"+str(run_id1)+"_id2_"+str(run_id2)+".log","w")
    f.write("DONE!")
    f.close()



