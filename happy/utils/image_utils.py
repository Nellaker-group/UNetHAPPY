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
    # returns in the image as a uin8 (to be converted to float32 later)
    return img


def load_image(image_path):
    img = skimage.io.imread(image_path)
    return process_image(img)
