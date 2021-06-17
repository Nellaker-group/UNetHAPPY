import numpy as np
import pandas as pd
import cv2


def annot_to_regular_boxes(annot_path, save_path, box_width, box_height):
    df = pd.read_csv(annot_path, names=["path", "x1", "y1", "x2", "y2", "class"])
    centre_xs = ((df["x2"] - df["x1"]) / 2) + df["x1"]
    centre_ys = ((df["y2"] - df["y1"]) / 2) + df["y1"]
    new_x1 = (centre_xs - box_width).astype(int)
    new_y1 = (centre_ys - box_height).astype(int)
    new_x2 = (centre_xs + box_width).astype(int)
    new_y2 = (centre_ys + box_height).astype(int)
    df["x1"] = new_x1
    df["y1"] = new_y1
    df["x2"] = new_x2
    df["y2"] = new_y2
    df.to_csv(save_path, header=False, index=False)


def draw_centre(img, x1, y1, x2, y2, label_name, cell=True):
    x = int((x2 + x1) / 2)
    y = int((y2 + y1) / 2)

    if not cell:
        cv2.circle(img, (x, y), 5, (255, 182, 109), 3)
    else:
        label_coords = (int(x) - 15, int(y) - 10)
        colour = _labels_and_colours(img, label_name, label_coords)
        cv2.circle(img, (x, y), 5, colour, 3)


def draw_box(img, x1, y1, x2, y2, label_name, cell=True):
    box = np.array((x1, y1, x2, y2)).astype(int)

    if not cell:
        cv2.rectangle(img, (x1, y1), (x2, y2), color=(255, 182, 109), thickness=2)
    else:
        label_coords = (box[0], box[1] - 10)
        colour = _labels_and_colours(img, label_name, label_coords)
        cv2.rectangle(img, (x1, y1), (x2, y2), color=colour, thickness=2)


def _labels_and_colours(img, label_name, label_coords):
    font = cv2.FONT_HERSHEY_PLAIN
    cv2.putText(img, label_name, label_coords, font, 1, (0, 0, 0), 2)
    cv2.putText(img, label_name, label_coords, font, 1, (255, 255, 255), 1)
    if label_name == "CYT":
        colour = (36, 255, 36)
    elif label_name == "FIB":
        colour = (0, 0, 146)
    elif label_name == "HOF":
        colour = (109, 255, 255)
    elif label_name == "SYN":
        colour = (255, 182, 109)
    elif label_name == "VEN":
        colour = (0, 150, 255)
    return colour
