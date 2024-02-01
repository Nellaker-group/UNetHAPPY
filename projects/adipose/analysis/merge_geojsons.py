from shapely.geometry import Point, Polygon, MultiPolygon, shape
import geojson
import typer

import projects.adipose.db.eval_runs_interface as db
import projects.adipose.data.merge_polygons as mp
from projects.adipose.data.geojsoner import  writeToGeoJSON, geojson2polygon, readGeoJSON2list


def main(
        geojson_file1: str = typer.Option(...),
        geojson_file2: str = typer.Option(...),
):
    geojson1 = readGeoJSON2list(geojson_file1)
    geojson2 = readGeoJSON2list(geojson_file2)
    geojson_list = [geojson1,geojson2]
    geojson_list_flat = [x for xs in geojson_list for x in xs]
    merged_polys_list = mp.merge_polysV3(geojson_list_flat)
    new_geojson_file1 = geojson_file1.split("/")[-1].replace(".geojson","")
    new_geojson_file2 = geojson_file2.split("/")[-1]
    new_geojson_name = f"{new_geojson_file1}_{new_geojson_file2}"
    writeToGeoJSON(merged_polys_list,new_geojson_name)

if __name__ == "__main__":
    typer.run(main)
