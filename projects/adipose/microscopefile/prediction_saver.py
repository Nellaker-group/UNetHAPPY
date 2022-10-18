"""
Class for saving model predictions on a whole slide image (WSI).
Data is saved and read from a DB by the public interface.
Saves coordinates which match the original WSI.
This means predictions are scaled down.
Before getting predictions, coords are scaled up to match model pixel sizes.

Public functionality (listed in order) includes:
- saving empty tiles
- saving nuclei from tile data and box predictions
- removing nuclei clusters from overlapped model predictions
- saving these valid nuclei into final storage
- saving cell classification at (x,y) from model predictions

Parameters:
file: MicroscopeFile object
"""
import numpy as np
import sklearn.neighbors as sk
from PIL import Image                                      # (pip install Pillow)
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

    # Saves segmentation as a polygon, where it is stored as a string of a list of tuples with the coordinates
    def save_seg(self, tile_index, polygons):
        tile_x = self.file.tile_xy_list[tile_index][0]
        tile_y = self.file.tile_xy_list[tile_index][1]        
        polyID=0
        coords = []
        if len(polygons) == 0:
            db.mark_finished_tiles(self.id, [tile_index])
        else:
            for poly in polygons:
                items = {}
                ## if it is a standalone polygon then it will have the type "Polygon" if it is a merged polygon, it will have the type "MultiPolygon"
                ## returns a list of tuples with each (x,y)
                if poly.type == 'Polygon':
                    ## extract x and y of points along edge
                    items["polyXY"] = str([(x+tile_x,y+tile_y) for x,y in poly.exterior.coords])
                    items["polyID"] = polyID
                if poly.type == 'MultiPolygon':
                    coordslist = [x.exterior.coords for x in poly.geoms]
                    tmpcoordslist=[x for xs in coordslist for x in xs]
                    items["polyXY"] = str([(x+tile_x,y+tile_y) for x,y in tmpcoordslist])
                    items["polyID"] = polyID
                coords.append(items)
                polyID += 1
                ## coords will be a string of a list of lists in this case                
        db.save_pred_workings(self.id, coords, tile_index)
        db.mark_finished_tiles(self.id, [tile_index])

    # emil - should this function be here? This file stores the polygons after drawn on the mask
    def draw_polygons_from_mask(self, mask, tile_index):
        w,h=np.shape(mask[0])    
        # emil
        padded_mask=np.zeros((w+2,h+2),dtype="uint8")    
        padded_mask[1:(w+1),1:(h+1)] = mask[0]
        # Find contours (boundary lines) around each sub-mask
        # Note: there could be multiple contours if the object
        # is partially occluded. (E.g. an elephant behind a tree)
        contours = measure.find_contours(padded_mask, 0.5, positive_orientation="low")  
        polygons = []
        segmentations = []
        for contour in contours:
            # Flip from (row, col) representation to (x, y)
            # and subtract the padding pixel
            # Emil has added X and Y coordinates to get global WSI coordinates
            for i in range(len(contour)):
                row, col = contour[i]
                contour[i] = (col - 1, row - 1)
            # Make a polygon and simplify it
            poly = Polygon(contour)
            poly = poly.simplify(1.0, preserve_topology=False)
            if(poly.is_empty):
                # Go to next iteration, dont save empty values in list
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



    def apply_seg_post_processing(self, overlap=True):
        seg_preds = db.get_all_unvalidated_seg_preds(self.id)
        seg_list = []

        for seg in seg_preds:            
            poly=Polygon([(x,y) for x,y in db.stringListTuple2coordinates(seg)])
            seg_list.append(poly)                

        merged_polys_list = []
        if overlap:            
            for poly in seg_list:
                merged_polys_list = mp.merge_polys(poly, merged_polys_list)

        merged_coords = []
        polyID=0
        for poly in merged_polys_list:
            items = {}
            if poly.type == 'Polygon':
                ## extract x and y of points along edge
                items["polyXY"] = str([(x,y) for x,y in poly.exterior.coords])
                items["polyID"] = polyID
            if poly.type == 'MultiPolygon':
                coordslist = [x.exterior.coords for x in poly.geoms]
                tmpcoordslist=[x for xs in coordslist for x in xs]
                items["polyXY"] = str([(x,y) for x,y in tmpcoordslist])
                items["polyID"] = polyID
            merged_coords.append(items)
            polyID += 1

        #nuclei_preds = self.cluster_multi_detections(nuclei_preds)
        # and perhaps filter off too small and too big ones
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



