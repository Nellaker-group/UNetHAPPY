from pathlib import Path
from enum import Enum

import typer
import pandas as pd
import cv2
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from nucnet.data.transforms.collaters import collater
from nucnet.data.dataset.nuclei_dataset import NucleiDataset
from nucnet.data.samplers.samplers import AspectRatioBasedSampler
from nucnet.data.transforms.transforms import Normalizer, Resizer, untransform_image
from nucnet.models import retinanet
from nucnet.utils.utils import print_gpu_stats, load_weights
from nucnet.data.utils import draw_box, draw_centre
from nucnet.microscopefile.prediction_saver import PredictionSaver
from nucnet.utils.enum_args import OrganArg

print_gpu_stats()


class ShapeArg(str, Enum):
    box: "box"
    point: "point"


def main(
    annot_path: str = typer.Option(...),
    csv_classes: str = typer.Option(...),
    pre_trained: str = typer.Option(...),
    shape: ShapeArg = ShapeArg.point,
    organ: OrganArg = OrganArg.placenta,
):
    dataset_name = "towards"
    score_thresh = 0.2

    dataset = NucleiDataset(
        annotations_dir=annot_path,
        dataset_names=[dataset_name],
        class_list_file=csv_classes,
        split="val",
        transform=transforms.Compose([Normalizer(), Resizer()]),
    )
    print("Dataset configured")

    sampler = AspectRatioBasedSampler(dataset, batch_size=1, drop_last=False)
    dataloader = DataLoader(
        dataset,
        num_workers=3,
        collate_fn=collater,
        batch_sampler=sampler,
        shuffle=False,
    )
    print("Dataloaders configured")

    model = retinanet.build_retina_net(
        num_classes=dataset.num_classes(), pretrained=False, resnet_depth=101
    )

    state_dict = torch.load(pre_trained)
    model = load_weights(state_dict, model)
    model = model.cuda()
    model = torch.nn.DataParallel(model).cuda()
    model.eval()
    print("Pushed model to cuda")

    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            scale = data["scale"]

            scores, _, boxes = model(data["img"].cuda().float())
            scores = scores.cpu().numpy()
            boxes = boxes.cpu().numpy()
            boxes /= scale

            filtered_preds = PredictionSaver.filter_by_score(
                150, score_thresh, scores, boxes
            )

            img = untransform_image(data["img"][0])

            save_dir = (
                Path(__file__).parent.parent
                / "projects"
                / organ.value
                / "visualisations"
                / "nuclei"
                / f"{dataset_name}_pred"
            )
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"val_{idx}.png"

            if len(filtered_preds) != 0:
                all_predictions = pd.DataFrame(
                    filtered_preds, columns=["x1", "y1", "x2", "y2"]
                )

                for i in range(len(all_predictions.index)):
                    x1 = all_predictions["x1"][i]
                    y1 = all_predictions["y1"][i]
                    x2 = all_predictions["x2"][i]
                    y2 = all_predictions["y2"][i]
                    label_name = "nucleus"

                    if shape.value == "point":
                        draw_centre(img, x1, y1, x2, y2, label_name, cell=False)
                    elif shape.value == "box":
                        draw_box(img, x1, y1, x2, y2, label_name, cell=False)
                    else:
                        raise ValueError(f"No such draw shape {shape.value}")

                print(f"saving to: {save_path}")
            else:
                print(f"no predictions on val_{idx}.png")

            cv2.imwrite(save_path, img)


if __name__ == "__main__":
    typer.run(main)
