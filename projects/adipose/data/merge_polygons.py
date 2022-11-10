import pandas as pd
from shapely.ops import unary_union
from shapely.geometry import Polygon

# old original version of the polygon merger
def merge_polys(new_poly, all_polys):
    all_polys_list = []
    for existing_poly in all_polys:
        if new_poly.intersects(existing_poly):
            new_poly = unary_union([new_poly, existing_poly])
        else:
            all_polys_list.append(existing_poly)
    all_polys_list.append(new_poly)
    return all_polys_list



# new faster version of the polygon merger - written by Phil and Chris
def merge_polysV2(polys, debug = False):

    bounds = pd.DataFrame([[*poly.bounds,False,-1] for poly in polys],columns=['min_x','min_y','max_x', 'max_y','checked','merge_start'])
    n_polys = bounds.shape[0]
    min_x_sorted = bounds.min_x.sort_values()
    min_y_sorted = bounds.min_y.sort_values()
    max_x_sorted = bounds.max_x.sort_values()
    max_y_sorted = bounds.max_y.sort_values()


    def get_shortlist_mask(poly):

        poly_min_x,poly_min_y,poly_max_x,poly_max_y = [*poly.bounds]

        #see https://stackoverflow.com/questions/31617845/how-to-select-rows-in-a-dataframe-between-two-values-in-python-pandas

        #a rectangle overlaps poly rectangle if all are true
        #min_x left of poly rect max_x
        min_x_lte_poly_max_x = pd.Series([False] * n_polys)
        min_x_lte_poly_max_x[min_x_sorted.index[:min_x_sorted.searchsorted(poly_max_x,side='right')]] = True
        #max_x right of poly rect min_x
        max_x_gte_poly_min_x = pd.Series([False] * n_polys)
        max_x_gte_poly_min_x[max_x_sorted.index[max_x_sorted.searchsorted(poly_min_x,side='left'):]] = True
        #min_y above poly rect max_y
        min_y_lte_poly_max_y = pd.Series([False] * n_polys)
        min_y_lte_poly_max_y[min_y_sorted.index[:min_y_sorted.searchsorted(poly_max_y,side='right')]] = True
        #max_y below poly rect min_y
        max_y_gte_poly_min_y = pd.Series([False] * n_polys)
        max_y_gte_poly_min_y[max_y_sorted.index[max_y_sorted.searchsorted(poly_min_y,side='left'):]] = True

        #do not return polys that don't overlap or have already been checked
        shortlist_mask = min_x_lte_poly_max_x & max_x_gte_poly_min_x & min_y_lte_poly_max_y & max_y_gte_poly_min_y & ~bounds.checked

        return shortlist_mask

    def find_and_merge(start_index,poly,report_recursion=0):
        if report_recursion:
            print('recursion depth %d' % report_recursion)
            report_recursion += 1
        original_bounds = poly.bounds
        for candidate_index in bounds.loc[get_shortlist_mask(poly),].index:
            if poly.intersects(polys[candidate_index]):
                bounds.loc[candidate_index,'checked'] = True
                if report_recursion:
                    print('merge %d into %d' % (candidate_index,start_index))
                bounds.loc[candidate_index,'merge_start'] = start_index
                poly = unary_union([poly,polys[candidate_index]])
        if poly.bounds != original_bounds: #only need to recurse if bounds have actually changed (complete subsumption won't change bounds)
            poly = find_and_merge(start_index,poly,report_recursion=report_recursion)
        return(poly)

    report_recursion = 0
    output_list = []
    for index in range(len(polys)):
        if ~bounds.loc[index,'checked']:
            if debug:
                print('poly %d' % index)
                report_recursion = 1
            bounds.loc[index,'checked'] = True
            bounds.loc[index,'merge_start'] = index
            output_list.append(find_and_merge(index,polys[index],report_recursion=report_recursion))
        elif debug:
            print('poly %d skipped (already checked)' % index)


    return(output_list,bounds)

def unit_test_merge_polysV2():

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
    poly_result,bounds_result = merge_polysV2([*test_polys.values()], debug = True)

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

    #affirm major merge poly is present and correct
    flag_merge_ok = False
    for curr_poly in poly_result:
        if curr_poly.bounds == expected_merged_poly.bounds:
            flag_merge_ok = True
    if flag_merge_ok:
        print('check ok\texpected merged poly')
    else:
        print('ERROR, expected clean poly list does not contain the expected merged poly')

    #affirm all polys were checked
    if bounds_result['checked'].all():
        print('check ok\tall polys considered for merging')
    else:
        print('ERROR, some polys were never considered for merging')

    #affirm correct merge sequence
    if expected_bounds_result['merge_start'].equals(bounds_result['merge_start']):
        print('check ok\texpected merging')
    else:
        print('ERROR, expected merge start list \n%s\ndoes not match result merge start list\n%s' %\
        ( '\n'.join(expected_bounds_result['merge_start'].astype(str).tolist()),'\n'.join(bounds_result['merge_start'].astype(str).tolist()) ))

    print('Completed unit tests of merge_poly')


