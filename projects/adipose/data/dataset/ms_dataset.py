import numpy as np

from happy.utils.utils import process_image
from happy.data.datasets.ms_dataset import MSDataset


class SegDataset(MSDataset):
    def _iter_data(self, iter_start, iter_end):
        for img, tile_index, empty_tile in self._get_dataset_section(        
            target_w=self.target_width,
            target_h=self.target_height,
            tile_range=(iter_start, iter_end),
        ):

            if not empty_tile:
                img = process_image(img).astype(np.float32) 
            sample = {
                "img": img,
                "tile_index": tile_index,
                "empty_tile": empty_tile,
                "scale": None,
                "annot": np.array([[0.0, 0.0, 0.0, 0.0, 0.0]]),
            }
            if self.transform and not empty_tile:
                sample = self.transform(sample)
            yield sample

    # Generator to create a dataset of tiles within a range
    def _get_dataset_section(self, target_w, target_h, tile_range):
        tile_coords = self.remaining_data[tile_range[0] : tile_range[1]]
        for _dict in tile_coords:
            img = self.file.get_tile_by_coords(
                _dict["tile_x"], _dict["tile_y"], target_w, target_h
            )
            if self.file._img_is_empty(img):
                yield None, _dict["tile_index"], True
            else:
                yield img, _dict["tile_index"], False
