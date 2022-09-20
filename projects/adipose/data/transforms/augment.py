import numpy as np
from scipy.ndimage import gaussian_filter
import scipy.stats as st
from scipy.ndimage.filters import gaussian_filter
import albumentations as A
import random

def albumentationAugmenter(image,mask,epochs):
    crop=random.choice(list(range(256,1024,2)))
    # inspired by Claudia's list of augs in ./projects/placenta/nuc_train.py
    # ReplayCompose, so that we can record which augmentations are used
    transform = A.ReplayCompose([
        #I cannot make CropAndPad work
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        #RandomBrigtnessContrast was causing problems distoring the augment images beyond reconigition - should be fixed now
        #tried setting brightness_by_maxBoolean to false (If True adjust contrast by image dtype maximum - and we have float32)
        A.RandomBrightnessContrast(p=0.25, brightness_limit=0.2, contrast_limit=0.2),
        A.Blur(blur_limit=5, p=0.25),
        #GaussNoise was causing problems distoring the augment images beyond reconigition - should be fixed now
        #after doing grid search and manually inspecting images, I chose var=0.01
        #A.GaussNoise(p=0.25,var_limit=(0.01, 0.01))
        #after advice from Chris I put it to 0.001
        A.GaussNoise(p=0.25,var_limit=(0.001, 0.001))
    ])
    transformed=transform(image=image, mask=mask)
    return(transformed['image'], transformed['mask'], transformed['replay'],1,crop)

