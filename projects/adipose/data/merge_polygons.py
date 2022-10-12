from shapely.ops import unary_union

def merge_polys(new_poly, all_polys):
    all_polys_list = []
    for existing_poly in all_polys:
        if new_poly.intersects(existing_poly):
            new_poly = unary_union([new_poly, existing_poly])
        else:
            all_polys_list.append(existing_poly)
    all_polys_list.append(new_poly)
    return all_polys_list

