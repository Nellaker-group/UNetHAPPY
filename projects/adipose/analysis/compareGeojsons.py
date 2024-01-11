import geopandas as gpd
from shapely.geometry import Point
import argparse

prs = argparse.ArgumentParser()
prs.add_argument('--geojson1', help="first geojson file (phil's file)", type=str)
prs.add_argument('--geojson2', help="second geojson file (strtree file)", type=str)
args = vars(prs.parse_args())


phil_gdf = gpd.read_file(args['geojson1'])
strtree_gdf = gpd.read_file(args['geojson2'])


#Can use a Point as index but can't hash it for set()
#so as a compromise, use the distance from the Point to (0,0)
phil_gdf.index = phil_gdf.representative_point().distance(Point(0, 0))
strtree_gdf.index = strtree_gdf.representative_point().distance(Point(0,0))


#Set logic to find the elements unique to each set
phil_gdf_ids = set(phil_gdf.index)
strtree_gdf_ids = set(strtree_gdf.index)
in_phil_not_strtree = phil_gdf.loc[list(phil_gdf_ids - strtree_gdf_ids),:]
in_strtree_not_phil = strtree_gdf.loc[list(strtree_gdf_ids - phil_gdf_ids),:]

#Write out filtered geojson
if in_phil_not_strtree.empty:
    print("in_geojson1_not_geojson2.geojson - is empty")
else:
    in_phil_not_strtree.to_file(args['geojson1'].replace('.geojson','in_geojson1_not_geojson2.geojson'), driver='GeoJSON')

if in_strtree_not_phil.empty:
    print("in_geojson2_not_geojson1.geojson - is empty")
else:
    in_strtree_not_phil.to_file(args['geojson2'].replace('.geojson','in_geojson2_not_geojson1.geojson'), driver='GeoJSON')

