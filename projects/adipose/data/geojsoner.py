from geojson import Point, Feature, FeatureCollection, dump
from shapely.geometry import Polygon, MultiPolygon

def writeToGeoJSON(masterList, filename):
    number=0
    features = []
    for poly in masterList:
        if poly.type == 'Polygon':
            features.append(Feature(geometry=Polygon([(x,y) for x,y in poly.exterior.coords]), properties={"id": str(number)}))
        if poly.type == 'MultiPolygon':
            mycoordslist = [list(x.exterior.coords) for x in poly.geoms]
            ll=[x for xs in mycoordslist for x in xs]
            features.append(Feature(geometry=Polygon(ll), properties={"id": str(number)}))
        number+=1
    feature_collection = FeatureCollection(features)
    ## I add another key to the FeatureCollection dictionary where it can look up max ID, so it knows which ID to work on when adding elements
    feature_collection["maxID"]=number
    with open(filename,"w") as outfile:
        dump(feature_collection, outfile)
    outfile.close()
