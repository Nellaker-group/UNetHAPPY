from shapely.geometry import Polygon


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


def db_to_list_within_tile(seg_preds,topX,topY):
    seg_list = []
    poly_list = []
    old_poly_id = seg_preds[0][0]
    tilePoly = Polygon([(topX, topY), (topX+1024, topY), (topX+1024, topY+1024), (topX, topY+1024)])
    for coord in seg_preds:
        if old_poly_id == coord[0]:
            poly_list.append((coord[2],coord[3]))
        else:
            poly=Polygon(poly_list)
            # checking polygon is valid
            if poly.is_valid and poly.within(tilePoly):
                seg_list.append(poly)
            poly_list = []
            # for the first point when poly_id changes
            poly_list.append((coord[2],coord[3]))
            old_poly_id = coord[0]
    return(seg_list)

