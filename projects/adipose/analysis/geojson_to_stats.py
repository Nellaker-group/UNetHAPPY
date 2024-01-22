import math
import statistics as stats
import geojson
from shapely.geometry import Polygon, MultiPolygon, shape
import numpy as np
import typer

import projects.adipose.db.eval_runs_interface as db
from projects.adipose.data.geojsoner import  writeToGeoJSON, geojson2polygon, readGeoJSON2list
from projects.adipose.analysis.get_pixel_size import get_pixel_size, get_which_pixel


def main(
    filename: str = typer.Option(..., help="Name of geojson file"),
    cohort : str = typer.Option(..., help="Name of geojson file"),
    write_geojson: bool = False, help="Whether to write geojson files, of all polygons, old filter polygons and new filter polygons",
):
    assert cohort in ["leipzig", "munich", "hohenheim", "endox", "gtex"]
    polys = readGeoJSON2list(filename)
    pp_all = []
    area_all = []
    polys_all = []
    area_old_filter = []
    polys_old_filter = []
    area_new_filter = []
    polys_new_filter = []
    pixel_size = get_pixel_size(cohort)
    for poly in polys:
        area_all.append(poly.area*pixel_size**2)
        pp_all.append((4*math.pi*poly.area ) / ((poly.length)**2))
        polys_all.append(poly)
        if poly.area*pixel_size**2 >= 200 and ((4*math.pi*poly.area ) / ((poly.length)**2)) > 0.75:
            area_old_filter.append(poly.area*pixel_size**2)
            polys_old_filter.append(poly)
        if poly.area*pixel_size**2 >= 10**2.5 and ((4*math.pi*poly.area ) / ((poly.length)**2)) > 0.6:
            area_new_filter.append(poly.area*pixel_size**2)        
            polys_new_filter.append(poly)
    print("ALL:")
    print("mean="+str(round(stats.mean(area_all),2))+", med="+str(round(stats.median(area_all),2))+", sd="+str(round(stats.stdev(area_all),2))+", N="+str(len(area_all)))
    print("After old filter (size > 200, PP > 0.75):")
    print("mean="+str(round(stats.mean(area_old_filter),2))+", med="+str(round(stats.median(area_old_filter),2))+", sd="+str(round(stats.stdev(area_old_filter),2))+", N="+str(len(area_old_filter)))
    print("After new filter (size > 316.23, PP > 0.6):")
    print("mean="+str(round(stats.mean(area_new_filter),2))+", med="+str(round(stats.median(area_new_filter),2))+", sd="+str(round(stats.stdev(area_new_filter),2))+", N="+str(len(area_new_filter)))
    if write_geojson:
        writeToGeoJSON(polys_all, filename.replace(".geojson","_all_polygons.geojson"))
        writeToGeoJSON(polys_old_filter, filename.replace(".geojson","_old_filter_polygons.geojson"))
        writeToGeoJSON(polys_new_filter, filename.replace(".geojson","_new_filter_polygons.geojson"))


if __name__ == "__main__":
    typer.run(main)

