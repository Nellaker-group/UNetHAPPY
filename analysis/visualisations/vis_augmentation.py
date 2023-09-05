import cv2
import albumentations as A

from happy.utils.utils import get_project_dir
from happy.data.transforms.utils.color_conversion import get_rgb_matrices
from happy.data.transforms.agumentations import StainAugment


def main():
    project_dir = get_project_dir("placenta")

    # read image
    img_path = str(project_dir / 'datasets/cell_class/triinV2/HOF/139_x23533_y16541.png')
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # define augmentation
    aug = A.Compose([
        # A.Flip(p=1.0),
        # A.RandomRotate90(p=1.0),
        StainAugment(get_rgb_matrices(), p=1.0, variance=0.4),
        # A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=1.0),
        # A.GaussNoise(var_limit=(500.0, 500.0), p=1.0),
        # A.Blur(blur_limit=(30,30), always_apply=True, p=1.0),
    ])

    # apply augmentation
    augmented = aug(image=image)
    augmented_image = augmented['image']

    # write result back to file
    output_path = str(project_dir / 'visualisations/cell_class/aug/output.png')
    augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, augmented_image)


if __name__ == "__main__":
    main()
