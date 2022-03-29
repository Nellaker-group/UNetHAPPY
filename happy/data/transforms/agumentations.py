import random

import albumentations as al
import numpy as np

from happy.data.transforms.utils.colorconv_he import he2rgb, rgb2he


class AlbAugmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(
        self,
        list_of_albumentations=[al.HorizontalFlip(p=0.5)],
        prgn=42,
        min_area=0.0,
        min_visibility=0.0,
        bboxes=True,
    ):
        self.list_of_albumentations = list_of_albumentations
        self.prgn = prgn
        self.min_area = min_area
        self.min_visibility = min_visibility
        self.bboxes = bboxes

    def __call__(self, sample):
        if self.bboxes:
            sample = self.bound_bbox_within_image(sample)
            alb_format = {
                "image": sample["img"],
                "bboxes": [x[:-1] for x in sample["annot"]],
                "category_id": [x[-1] for x in sample["annot"]],
            }
        else:
            alb_format = {"image": sample["img"], "category_id": sample["annot"]}

        if self.bboxes:
            alb_aug = al.Compose(
                self.list_of_albumentations,
                bbox_params={
                    "format": "pascal_voc",
                    "min_area": self.min_area,
                    "min_visibility": self.min_visibility,
                    "label_fields": ["category_id"],
                },
            )
        else:
            alb_aug = al.Compose(self.list_of_albumentations)
        alb_format = alb_aug(**alb_format)
        if self.bboxes:
            sample = {
                "img": alb_format["image"],
                "annot": np.array(
                    [
                        np.append(y, x)
                        for y, x in zip(
                            alb_format["bboxes"],
                            [[z] for z in alb_format["category_id"]],
                        )
                    ]
                ),
            }
        else:
            sample = {
                "img": alb_format["image"],
                "annot": sample["annot"],
            }
        return sample

    def bound_bbox_within_image(self, sample):
        im_y, im_x, im_channels = sample["img"].shape
        for xindex, x in enumerate(sample["annot"]):
            bbx_x1, bbx_y1, bbx_x2, bbx_y2, bbx_cat = x
            assert bbx_x1 < bbx_x2
            assert bbx_y1 < bbx_y2
            if not bbx_x1 >= 0:
                bbx_x1 = 0
            if not bbx_y1 >= 0:
                bbx_y1 = 0
            if not bbx_x2 <= im_x:
                bbx_x2 = im_x
            if not bbx_y2 <= im_y:
                bbx_y2 = im_y
            sample["annot"][xindex] = [bbx_x1, bbx_y1, bbx_x2, bbx_y2, bbx_cat]
        return sample


class GaussNoise_Augment_stylealb(al.ImageOnlyTransform):
    """Apply gaussian noise to the input image. Modified to work within dtype range 0-1.
    Args:
        var_limit ((float, float) or float): variance range for noise. If var_limit is a single float, the range
            will be (-var_limit, var_limit). Default: (10., 50.).
        p (float): probability of applying the transform. Default: 0.5.
    Targets:
        image
    Image types:
        uint8
    """

    def __init__(self, var_limit=(0.1, 0.5), always_apply=False, p=0.5):
        super(GaussNoise_Augment_stylealb, self).__init__(always_apply, p)
        self.var_limit = al.core.transforms_interface.to_tuple(var_limit)

    def apply(self, img, gauss=None, **params):
        return al.augmentations.functional.gauss_noise(img, gauss=gauss)

    def get_params_dependent_on_targets(self, params):
        image = params["image"]
        var_max = np.max(image)
        var = var_max * random.uniform(self.var_limit[0], self.var_limit[1])
        mean = var
        sigma = var ** 0.5
        if var_max <= 1.0:
            sigma = var ** 2
        random_state = np.random.RandomState(random.randint(0, 2 ** 32 - 1))
        gauss = random_state.normal(mean, sigma, image.shape)
        gauss = gauss - np.min(gauss)
        return {"gauss": gauss}

    @property
    def targets_as_params(self):
        return ["image"]


class Stain_Augment_stylealb(al.ImageOnlyTransform):
    """Convert the input RGB image to HED (Heamatoxylin, Eosin, DAPI) vary the staining in H and E, return to RGB.
    Args:
        p (float): probability of applying the transform. Default: 0.5.
        variance (float): factor by which HE staining is randomly varied
    Targets:
        image
    Image types:
        uint8, float32
    """

    def __init__(self, variance=0.1, always_apply=False, p=0.5):
        super(Stain_Augment_stylealb, self).__init__(always_apply, p)
        self.variance = variance

    def apply(self, img, variance=0.1, **params):
        img = rgb2he(img * 255.0)  # mod for this augmentation, undone at last step

        # tweak Heamatoxylin
        img[:, :, [0]] = (
            np.random.uniform(low=-variance, high=variance) + img[:, :, [0]]
        )
        # tweak Eosin
        img[:, :, [1]] = (
            np.random.uniform(low=-variance, high=variance) + img[:, :, [1]]
        )

        img = he2rgb(img)
        return img / 255.0

    def get_params(self):
        return {"variance": self.variance}
