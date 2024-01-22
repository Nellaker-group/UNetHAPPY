from shapely.geometry import Polygon
import pickle
import typer

import projects.adipose.db.eval_runs_interface as db
from projects.adipose.data.geojsoner import  writeToGeoJSON
import projects.adipose.data.merge_polygons as mp


def main(pickle_file: str = typer.Option(...), merge: bool = True):
    file = open(pickle_file,'rb')
    poly_preds = pickle.load(file)
    file.close()
    poly_list = []
    for poly in poly_preds:
        poly_list.append(poly)
    geojsonFile = pickle_file.replace(".obj",".geojson")
    writeToGeoJSON(poly_list, geojsonFile)
    if merge:            
        merged_polys_list = []
        merged_polys_list = mp.merge_polysV3(poly_list)
        geojsonFile2 = pickle_file.replace(".obj","_merged.geojson")
        writeToGeoJSON(merged_polys_list, geojsonFile2)

if __name__ == "__main__":
    typer.run(main)
