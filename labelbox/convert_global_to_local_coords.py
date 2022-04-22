import pandas as pd


def convert_global_to_local(annotation, subregion=False):
    image_name = annotation["image_name"]
    name_parts = image_name.split(".")[0].split("_")
    xmin = int(name_parts[1].split("x")[1])
    ymin = int(name_parts[2].split("y")[1])
    coordinates = annotation["coordinates"]

    try:
        downsample_factor = annotation["downsample"]
    except KeyError:
        print("Downsample factor being calculated.")
        downsample_factor = _calc_downsample_factor(coordinates)

    local_polygon_coords = []
    for point in coordinates:
        local_x, local_y = _global_coord_to_local(xmin, ymin, point["x"], point["y"])
        if not subregion:
            local_x += 250
            local_y += 250
        local_x = round(local_x / downsample_factor, 3)
        local_y = round(local_y / downsample_factor, 3)
        local_polygon_coords.append({"x": local_x, "y": local_y})

    return local_polygon_coords

# Only used if it isn't found in annotations. Kept for old annotation files.
def _calc_downsample_factor(coordinates):
    coordinates = pd.DataFrame(coordinates)
    xmin = coordinates["x"].min()
    xmax = coordinates["x"].max()
    ymin = coordinates["y"].min()
    ymax = coordinates["y"].max()
    width = xmax - xmin
    height = ymax - ymin
    if width >= 5000:
        return 4
    elif width <= 200:
        return 1
    elif height <= 200:
        return 1
    return 1.5

def _global_coord_to_local(xmin, ymin, x, y):
    if x == 0 and y == 0:
        return 0, 0
    if x == 0:
        return 0, y - ymin
    if y == 0:
        return x - xmin, 0
    return x - xmin, y - ymin