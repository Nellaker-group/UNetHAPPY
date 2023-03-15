import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon, shape
import argparse
import db.eval_runs_interface as db
from data.geojsoner import  writeToGeoJSON, geojson2polygon, readGeoJSON2list
import data.merge_polygons as mp
import geojson


prs = argparse.ArgumentParser()
prs.add_argument('--geojson1', help="first geojson file", type=str)
prs.add_argument('--geojson2', help="second geojson file", type=str)
args = vars(prs.parse_args())


g1 = readGeoJSON2list(args['geojson1'])
g2 = readGeoJSON2list(args['geojson2'])

gg = [g1,g2]

gg2 = [x for xs in gg for x in xs]

merged_polys_list = []
merged_polys_list = mp.merge_polysV3(gg2)



writeToGeoJSON(merged_polys_list,args['geojson1'].replace(".geojson","")+"_"+args['geojson2'])
