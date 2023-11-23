"""
Class for saving model predictions on a whole slide image (WSI).
Data is saved and read from a DB by the public interface.
Saves coordinates of polygons which match the original WSI.
This means predictions are scaled down.
Before getting predictions, coords are scaled up to match model pixel sizes.

Public functionality (listed in order) includes:
- saving empty tiles
- saving polygons from segmentations from tile data
- merging overlapping polygons
- saving these merged polygons into final storage

Parameters:
file: MicroscopeFile object
"""
import numpy as np                                         # (pip install numpy)
from skimage import measure                                # (pip install scikit-image)
from shapely.geometry import Polygon, MultiPolygon, shape  # (pip install Shapely)

import db.eval_runs_interface as db
import data.geojsoner as gj
import data.merge_polygons as mp


class PredictionSaver:
    def __init__(self, microscopefile):
        self.file = microscopefile
        self.id = self.file.id
        self.rescale_ratio = self.file.rescale_ratio


    # Saving tiles which do not have predictions as caught by pixel colours
    def save_empty(self, tile_indexes):
        db.mark_finished_tiles(self.id, tile_indexes)


    # Saves segmentation as a polygon, where it stores X and Y coordinates for each point along with poly_id
    def save_seg(self, tile_index, polygons, latest_poly_id):
        tile_x = self.file.tile_xy_list[tile_index][0]
        tile_y = self.file.tile_xy_list[tile_index][1]        
        coords = []
        # for keeping track of current polygons, latest_poly_id keeps track of polygons up until now!
        poly_id = 0
        if len(polygons) == 0:
            db.mark_finished_tiles(self.id, [tile_index])
        else:
            for poly in polygons:
                point_id = 0
                # it appends a tuple for each point with poly_id and X and Y coordinate - casting them to ints
                if poly.type == 'Polygon':
                    # because otherwise the starting/end point is included twice
                    for x,y in poly.exterior.coords[:-1]:
                        x_scaled = x*self.rescale_ratio+tile_x
                        y_scaled = y*self.rescale_ratio+tile_y
                        coords.append((poly_id,point_id,int(x_scaled),int(y_scaled)))
                        point_id += 1
                if poly.type == 'MultiPolygon':
                    coordslist = [x.exterior.coords[:-1] for x in poly.geoms]
                    tmpcoordslist=[x for xs in coordslist for x in xs]
                    for x,y in tmpcoordslist:
                        x_scaled = x*self.rescale_ratio+tile_x
                        y_scaled = y*self.rescale_ratio+tile_y
                        coords.append((poly_id,point_id,int(x_scaled),int(y_scaled)))
                        point_id += 1
                poly_id += 1
        db.save_pred_workings(self.id, coords, latest_poly_id)
        db.mark_finished_tiles(self.id, [tile_index])


    def draw_polygons_from_mask(self, mask, tile_index):
        w,h=np.shape(mask[0])    
        padded_mask=np.zeros((w+2,h+2),dtype="uint8")    
        padded_mask[1:(w+1),1:(h+1)] = mask[0]
        # Find contours (boundary lines) around each sub-mask
        # Note: there could be multiple contours if the object
        # is partially occluded. (E.g. an elephant behind a tree)
        contours = measure.find_contours(padded_mask, 0.5, positive_orientation="low")  
        polygons = []
        segmentations = []
        for contour in contours:
            # Flip from (row, col) representation to (x, y) and subtract the padding pixel - added X and Y coordinates to get global WSI coordinates
            for i in range(len(contour)):
                row, col = contour[i]
                contour[i] = (col - 1, row - 1)
            # Make a polygon and simplify it - simplification makes it run much faster
            poly = Polygon(contour)
            poly = poly.simplify(1.0, preserve_topology=False)
            if(poly.is_empty):
                # Go to next iteration, don't save empty values in list
                continue
            if(not poly.is_valid):
                # Go to next iteration, don't save invalid polygon
                continue
            polygons.append(poly)
        # checking that polygons are not contained in another polygon
        polygonsKeep = []
        for j in range(0, len(polygons)):                
            contained=False
            intersected=False
            for i in range(0, len(polygons)):
                if polygons[j].contains(polygons[i]) and i != j:                
                    contained=True
                if polygons[j].intersects(polygons[i]) and i != j:
                    intersected=True
            if contained and intersected:
                polygonsKeep.append(polygons[j])
            elif not intersected and not contained:
                polygonsKeep.append(polygons[j])
        return polygonsKeep


    def apply_seg_post_processing(self, write_geojson, overlap=True):
        seg_preds = db.get_all_unvalidated_seg_preds(self.id)
        seg_list = []
        run = db.get_eval_run_by_id(self.id)
        poly_list = []
        old_poly_id = seg_preds[0][0]
        # it runs through all the points and checks if it is a new polygon by checking if poly_id has changed
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
        print("list of polygons pre merge is this long:")
        print(len(seg_list))
        print("self.file.slide_path")
        print(self.file.slide_path)
        # merges overlapping polygons, to have a list of merged non overlapping polygons
        merged_polys_list = []
        if overlap:            
            merged_polys_list = mp.merge_polysV3(seg_list)
        slideName = self.file.slide_path.split("/")[-1].split(".")[0]
        if write_geojson:
            gj.writeToGeoJSON(merged_polys_list, slideName+'coords_merged_segmodel'+str(run.seg_model)+'_overlap'+str(run.overlap)+'_px'+str(run.pixel_size)+'.geojson')
        merged_coords = []
        poly_id=0        
        for poly in merged_polys_list:            
            point_id = 0
            # it updates with a new poly_id for the merged polygons
            if poly.type == 'Polygon':
                for x,y in poly.exterior.coords:
                    merged_coords.append((poly_id,point_id,int(x),int(y)))
                    point_id += 1
            if poly.type == 'MultiPolygon':
                coordslist = [x.exterior.coords for x in poly.geoms]
                tmpcoordslist=[x for xs in coordslist for x in xs]
                for x,y in tmpcoordslist:
                    merged_coords.append((poly_id,point_id,int(x),int(y)))
                    point_id += 1
            poly_id += 1        
        # push the predictions to the database
        db.validate_pred_workings(self.id, merged_coords)
        self.file.mark_finished_seg()


    # Inserts valid/non duplicate predictions into Predictions table
    def commit_valid_seg_predictions(self):
        db.commit_pred_workings(self.id)

    @staticmethod
    def filter_by_score(threshold, scores):
        flatScores = scores[0]
        flatScores[flatScores > threshold] = 255
        flatScores[flatScores <= threshold] = 0
        return(flatScores)



