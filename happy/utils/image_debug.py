import cv2
import sys
import os
from pathlib import Path

import numpy as np
from PIL import Image

from happy.data.transforms.transforms import UnNormalizer, unnormalise_image


def test_images(data, iter_num, save_path):
    unnormalize = UnNormalizer()
    img = np.array(255 * unnormalize(data["img"][0, :, :, :])).copy()
    img[img < 0] = 0
    img[img > 255] = 255

    img = np.transpose(img, (1, 2, 0))
    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
    im = Image.fromarray(img.astype("uint8"))

    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    im.save(save_dir / f"image_test_{iter_num}.png")


def debug_augmentations(dataloader_train):
    debug_dir = Path("debug_dir")
    debug_dir.mkdir(parents=True, exist_ok=True)
    if not os.path.exists(debug_dir):
        os.mkdir(debug_dir)
    for iter_num, data in enumerate(dataloader_train):
        img = unnormalise_image(data["img"][0])
        img.save(debug_dir / f"aug_{iter_num}_img.jpg")
    sys.exit(1)
