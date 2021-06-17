import skimage.color
import skimage.io
import numpy as np


def process_image(img):
    if img.shape[2] == 2:
        img = skimage.color.gray2rgb(img)
    elif img.shape[2] == 4:
        img = np.array(img)[:, :, 0:3]
    elif img.shape[2] == 3:
        pass
    else:
        raise ValueError(f"Unexpected number of channels {img.shape[2]}")

    return img.astype(np.float32) / 255.0


# todo: try to remove this try except
def load_image(image_path):
    try:
        img = skimage.io.imread(image_path)
    except FileNotFoundError:
        img = skimage.io.imread("../" + image_path)

    return process_image(img)
