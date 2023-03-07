import pandas as pd
from shapely.ops import unary_union
from shapely.geometry import Polygon
from shapely.strtree import STRtree
from data.geojsoner import writeToGeoJSON


# new faster version of the polygon merger - written by Emil (with inspiration from Phil and Chris's code) using STRtree from shapely
def merge_polysV3(polys, debug = False):
    #shapely STRtree for searching through - returns a list of all geometries in the strtree whose extents intersect the extent of geom (poly)
    poly_tree = STRtree(polys)
    #for keeping track of what has been merged already
    merged=pd.Series([False for i in range(len(polys))])
    # for getting indexes of the polygons from STRtree (from shapely's website)
    index_by_id = dict((id(pt), i) for i, pt in enumerate(polys))
    def get_shortlist_mask(poly):
        #get all the candidate polygons using the STRtree
        shortlist_mask_tmp = poly_tree.query(poly)
        #getting the indexes of the queried polygons
        indexes = [index_by_id[id(pt)] for pt in shortlist_mask_tmp]
        #for keeping track of polygons in query
        counter = 0
        shortlist_mask = []
        #go through polygons and remove those already been merged
        for poly in shortlist_mask_tmp:
            #check if polygon has already been merged - because candidate polygons might have been merged - using the merged and the indexes of merged polygons
            if not merged[indexes[counter]]:
                shortlist_mask.append((poly,indexes[counter]))                     
            counter += 1
        return shortlist_mask
    #check each candidate poly if intersects,
    #recurses if box around target poly has grown - and find new candidates via get_shortlist_mask
    def find_and_merge(poly,report_recursion=0):
        if report_recursion:
            print('recursion depth %d' % report_recursion)
            report_recursion += 1
        original_bounds = poly.bounds
        #loop through candidates from STRtree (list of polygons)
        for candidate_poly, candidate_index in get_shortlist_mask(poly):
            if poly.intersects(candidate_poly):
                #marks that this polygon has been merged - because polygon once it is merged is no longer needed - the merged one is
                merged[candidate_index] = True
                if report_recursion:
                    print('merge %d' % (candidate_index))
                    # try and merge if there is an error just keep original polygon
                try:
                    poly = unary_union([poly,candidate_poly])
                except ValueError:
                    print("bad candidate poly, it is not being merged - candidate_index:")
                    print(candidate_index)
        if poly.bounds != original_bounds: #only need to recurse if bounds/boxes have actually changed (complete subsumption won't change bounds)
            #recurses on the function with the new merged polygon and the updated indexes of merged polygons
            poly = find_and_merge(poly,report_recursion=report_recursion)
        return(poly)
    report_recursion = 0
    output_list = []
    #it goes through all polygons
    for index in range(len(polys)):
        #if polygon has already been merged - do not bother - polygon once it is merged is no longer needed - the merged one is
        if not merged[index]:
            if debug:
                print('poly %d' % index)
                report_recursion = 1
            merged[index] = True
            poly = find_and_merge(polys[index],report_recursion=report_recursion)            
            # because it might return some empty polygons that will mess up the geojson file
            if not poly.is_empty:
                output_list.append(poly)
        elif debug:
            print('poly %d skipped (already checked)' % index)
    return(output_list)



############################################################
############################################################


def unit_test_merge_polysV3():
    test_polys = {
    'target_poly': Polygon([(5,5),(5,10),(10,10),(10,5)]),
    'overlap_poly_1':  Polygon([(1,1),(1,6),(6,6),(6,1)]), #top left corner
    'overlap_poly_2':  Polygon([(7,1),(7,6),(15,6),(15,1)]), #top right corner
    'overlap_poly_3':  Polygon([(1,8),(1,13),(6,13),(6,8)]), #lower left corner
    'overlap_poly_4':  Polygon([(9,9),(9,12),(12,12),(12,9)]), #lower right corner
    'no_over_poly_near':  Polygon([(7,11),(7,13),(8,13),(8,11)]), #no overlap below but near
    'no_over_poly_far':  Polygon([(35,1),(35,3),(37,3),(37,1)]), #no overlap right very far
    'outer_fulloverlap_poly':  Polygon([(35,10),(35,15),(38,15),(38,10)]), #fully overlaps with inner_fulloverlap_poly
    'inner_fulloverlap_poly':  Polygon([(36,11),(36,14),(37,14),(37,11)]), #fully overlaps with inner_fulloverlap_poly
    }
    #construct expected output
    #expected fully merged polygon
    expected_merged_poly = Polygon([(1,1),(1,6),(5,6),(5,8),(1,8),(1,13),(6,13),(6,10),(9,10),(9,12),(12,12),(12,9),(10,9),(10,6),(15,6),(15,1),(7,1),(7,5),(6,5),(6,1)])
    #expected set of polygons
    expected_poly_result = [expected_merged_poly, test_polys['no_over_poly_near'], test_polys['no_over_poly_far'], test_polys['outer_fulloverlap_poly']]
    #expected bounds df
    expected_bounds_result = pd.DataFrame(\
        [[*poly.bounds,True,merge_start] for (poly,merge_start) in [
            (test_polys['target_poly'], 0),
            (test_polys['overlap_poly_1'], 0),
            (test_polys['overlap_poly_2'], 0),
            (test_polys['overlap_poly_3'], 0),
            (test_polys['overlap_poly_4'], 0),
            (test_polys['no_over_poly_near'], 5),
            (test_polys['no_over_poly_far'], 6),
            (test_polys['outer_fulloverlap_poly'], 7),
            (test_polys['outer_fulloverlap_poly'], 7)]]
        ,columns=['min_x', 'min_y','max_x', 'max_y','checked','merge_start'])
    #run test with debug active
    poly_result = merge_polysV3([*test_polys.values()], debug = True)
    #affirm correct clean list
    num_singular_matches = 0
    num_multi_matches = 0
    num_missing = 0
    for expected_poly in expected_poly_result:
        matches = 0
        for poly in poly_result:
            if expected_poly.equals(poly):
                matches += 1
        if matches == 1:
            num_singular_matches += 1
        elif matches > 1:
            num_multi_matches += 1
        else:
            num_missed_matches = 0
    if (num_singular_matches == len(expected_poly_result)) and(num_singular_matches == len(poly_result)) and (num_multi_matches == 0) and (num_missing == 0):
        print('check ok\tclean_poly list')
    else:
        print('''\
ERROR, expected clean poly list and returned poly list are not same. Length expected %d vs returned %d
number of 1:1 matches %d; number of 1:many matches %d; number of missed matches %d''' %\
        (len(expected_poly_result), len(poly_result), num_singular_matches, num_multi_matches, num_missed_matches))


