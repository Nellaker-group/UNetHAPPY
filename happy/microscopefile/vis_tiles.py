import cv2
import numpy as np
from PIL import Image

from happy.db.msfile_interface import get_msfile_by_run
from happy.microscopefile.utils import get_nuc_locs


# Saves an image of the tile with nuclei detections
def vis_nuclei_on_tile(
    run_id, save_path, tile_coords, tile_width=1600, tile_height=1200
):
    file = get_msfile_by_run(run_id)
    min_x = tile_coords[0]
    min_y = tile_coords[1]

    nuc_preds = get_nuc_locs(run_id, file, min_x, min_y, tile_width, tile_height)
    print(f"number of predictions: {len(nuc_preds)}")
    img_array = file.get_tile_by_coords(
        tile_coords[0], tile_coords[1], tile_width, tile_height
    )

    im = Image.fromarray(img_array.astype("uint8"))
    draw = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

    for (x, y, _) in nuc_preds:
        x = x - min_x
        y = y - min_y

        x_rescale = (x / file.rescale_ratio).astype("float32")
        y_rescale = (y / file.rescale_ratio).astype("float32")

        cv2.circle(draw, (x_rescale, y_rescale), 5, (50, 255, 255), 3)

    slide_name = file.slide_path.split("/")[-1]
    name = f"{save_path}/{slide_name.split('-')[0]}_x{min_x}_y{min_y}_nucvis.png"
    cv2.imwrite(name, draw)


# Saves an image of the tile with cell classifications
def vis_cells_on_tile(
    run_id, save_path, tile_coords, tile_width=1600, tile_height=1200
):
    file = get_msfile_by_run(run_id)
    min_x = tile_coords[0]
    min_y = tile_coords[1]

    nuc_preds = get_nuc_locs(run_id, file, min_x, min_y, tile_width, tile_height)
    print(f"number of predictions: {len(nuc_preds)}")

    labels = {0: "CYT", 1: "FIB", 2: "HOF", 3: "SYN", 4: "VEN"}
    img_array = file.get_tile_by_coords(
        tile_coords[0], tile_coords[1], tile_width, tile_height
    )

    img = Image.fromarray(img_array.astype("uint8"))
    draw = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    for i, (x, y, cell_class) in enumerate(nuc_preds):
        x = x - min_x
        y = y - min_y

        x_rescale = (x / file.rescale_ratio).astype("float32")
        y_rescale = (y / file.rescale_ratio).astype("float32")

        cv2.putText(
            draw,
            labels[cell_class],
            (int(x_rescale) - 15, int(y_rescale) - 10),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (0, 0, 0),
            2,
        )
        cv2.putText(
            draw,
            labels[cell_class],
            (int(x_rescale) - 15, int(y_rescale) - 10),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (255, 255, 255),
            1,
        )

        if cell_class == 0:
            cv2.circle(draw, (x_rescale, y_rescale), 5, (36, 255, 36), 3)
        if cell_class == 1:
            cv2.circle(draw, (x_rescale, y_rescale), 5, (0, 0, 146), 3)
        if cell_class == 2:
            cv2.circle(draw, (x_rescale, y_rescale), 5, (109, 255, 255), 3)
        if cell_class == 3:
            cv2.circle(draw, (x_rescale, y_rescale), 5, (255, 182, 109), 3)
        if cell_class == 4:
            cv2.circle(draw, (x_rescale, y_rescale), 5, (0, 150, 255), 3)

    slide_name = file.slide_path.split("/")[-1]
    name = f"{save_path}/{slide_name.split('-')[0]}_x{min_x}_y{min_y}_cellvis.png"
    cv2.imwrite(name, draw)
