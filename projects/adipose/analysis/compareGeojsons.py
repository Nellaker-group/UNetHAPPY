import geopandas as gpd
from shapely.geometry import Point
import typer


def main(geojson1_filename: str = typer.Option(..., help="first geojson file (e.g. regular .geojson file"), geojson2_filename: str = typer.Option(..., help="second geojson file (e.g. strtee file)")):

    first_gdf = gpd.read_file(geojson1_filename)
    second_gdf = gpd.read_file(geojson2_filename)

    #Can use a Point as index but can't hash it for set()
    #so as a compromise, use the distance from the Point to (0,0)
    first_gdf.index = first_gdf.representative_point().distance(Point(0, 0))
    second_gdf.index = second_gdf.representative_point().distance(Point(0,0))

    #Set logic to find the elements unique to each set
    first_gdf_ids = set(first_gdf.index)
    second_gdf_ids = set(second_gdf.index)
    in_first_not_second = first_gdf.loc[list(first_gdf_ids - second_gdf_ids),:]
    in_second_not_first = second_gdf.loc[list(second_gdf_ids - first_gdf_ids),:]

    #Write out filtered geojson
    if in_first_not_second.empty:
        print("in_geojson1_not_geojson2.geojson - is empty")
    else:
        in_first_not_second.to_file(geojson1_filename.replace('.geojson','in_geojson1_not_geojson2.geojson'), driver='GeoJSON')

    if in_second_not_first.empty:
        print("in_geojson2_not_geojson1.geojson - is empty")
    else:
        in_second_not_first.to_file(geojson2_filename.replace('.geojson','in_geojson2_not_geojson1.geojson'), driver='GeoJSON')


if __name__ == "__main__":
    typer.run(main)

