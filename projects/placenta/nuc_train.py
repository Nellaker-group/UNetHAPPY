import argparse
import collections
from pathlib import Path
from datetime import datetime

import albumentations as al
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import transforms

from happy.train.od_training_eval import evaluate
from happy.data.transforms.agumentations import (
    AlbAugmenter,
    Stain_Augment_stylealb,
    GaussNoise_Augment_stylealb,
)
from happy.data.transforms.collaters import collater
from happy.data.dataset.nuclei_dataset import NucleiDataset
from happy.data.samplers.samplers import AspectRatioBasedSampler
from happy.data.transforms.transforms import Normalizer, Resizer
from happy.utils.hyperparameters import Hyperparameters
from happy.utils.image_debug import debug_augmentations
from happy.models import retinanet
from happy.utils.utils import print_gpu_stats, load_weights
from happy.utils.vis_plotter import VisdomLinePlotter

use_gpu = True  # a debug flag to allow non GPU testing of stuff. default true

if use_gpu:
    print_gpu_stats()


def main():
    """
    Define parser variables and hard coded constant variables.
    Can either load parameters from a file, add them yourself, or both.

    Example use without loading from params file:
    python nuc_train.py --exp_name exp_1 --annot_dir Annotations/Nuclei/ --dataset_names triin towards empty
    --csv_classes Annotations/Nuclei/classes.csv --pre_trained None --epochs 5 --batch 2 --learning_rate 1e-5
    --init_from_coco True --vis True

    Example use loading from params file:
    python nuc_train.py --exp_name exp_1 --params_load_path Results/Nuclei/exp_1/default_params.csv

    Example use loading from params file and changing a variable:
    python nuc_train.py --exp_name exp_1 --params_load_path Results/Nuclei/exp_1/default_params.csv --epochs 100
    """

    parser = argparse.ArgumentParser(
        description="Simple training script for training a RetinaNet network."
    )
    parser.add_argument(
        "--params_load_path",
        help="Path to csv file containing hyperparams and arguments.",
    )
    parser.add_argument(
        "--exp_name", help="Name of the experiment type you are running.", required=True
    )
    parser.add_argument(
        "--annot_dir", help="Path to directory containing dataset-specific annotations"
    )
    parser.add_argument(
        "--dataset_names",
        help="list of dataset names to be used for training and eval",
        nargs="+",
    )
    parser.add_argument("--csv_classes", help="Path to file containing class list")
    parser.add_argument(
        "--pre_trained", help="Path to file containing pretrained weights"
    )
    parser.add_argument("--epochs", help="Number of epochs", type=int)
    parser.add_argument("--batch", help="Batch size", type=int)
    parser.add_argument("--learning_rate", help="Learning rate, e.g. 1e-5", type=float)
    parser.add_argument(
        "--init_from_coco", help="Initialise training from coco weights?"
    )
    parser.add_argument("--vis", help="Activate visdom visualisations")
    debugging_flag = False

    args = parser.parse_args()

    # Loads from file and overwrites with user inputs
    if args.params_load_path:
        hp = Hyperparameters.load_from_csv(f"../{args.params_load_path}")
        hp.resolve_parser_overwrites(args)
    else:
        hp = Hyperparameters(
            args.exp_name,
            args.annot_dir,
            args.dataset_names,
            args.csv_classes,
            args.pre_trained,
            args.epochs,
            args.batch,
            args.learning_rate,
            args.init_from_coco,
            args.vis,
        )

    multiple_val_sets = True if len(hp.dataset_names) > 1 else False

    """
    Defining the Visdom visualisations (make sure the ports are tunneling appropriately)
    
    """
    if hp.vis == "True":
        vis = VisdomLinePlotter()

    """ 
    Define data sets for both train and val. This also defines the augmentations that will be used.
    """

    # Create the training dataset from all directories specified in dataset_names
    train_dataset = NucleiDataset(
        annotations_dir=f"../{hp.annot_dir}",
        dataset_names=hp.dataset_names,
        class_list_file=f"../{hp.csv_classes}",
        split="train",
        transform=transforms.Compose(
            [
                AlbAugmenter(
                    list_of_albumentations=[
                        al.Flip(p=0.9),
                        al.RandomRotate90(p=0.9),
                        Stain_Augment_stylealb(p=0.9, variance=0.4),
                        GaussNoise_Augment_stylealb(var_limit=(0.05, 0.2), p=0.85),
                        GaussNoise_Augment_stylealb(var_limit=(0.01, 0.05), p=0.85),
                        GaussNoise_Augment_stylealb(var_limit=(0.01, 0.05), p=0.85),
                        al.Blur(blur_limit=5, p=0.8),
                        al.Rotate(limit=(0, 45), p=0.8),
                        al.RandomScale(scale_limit=0.05, p=0.8),
                    ]
                ),
                Normalizer(),
                Resizer(),
            ]
        ),
    )

    # Create the validation dataset from all directories specified in dataset_names
    all_val_dataset = NucleiDataset(
        annotations_dir=f"../{hp.annot_dir}",
        dataset_names=hp.dataset_names,
        class_list_file=f"../{hp.csv_classes}",
        split="val",
        transform=transforms.Compose([Normalizer(), Resizer()]),
    )

    # Create individual validation datasets from all directories specified in dataset_names
    if multiple_val_sets:
        dataset_val_dict = {}
        for dataset_name in hp.dataset_names:
            dataset_val_dict[dataset_name] = NucleiDataset(
                annotations_dir=f"../{hp.annot_dir}",
                dataset_names=dataset_name,
                class_list_file=f"../{hp.csv_classes}",
                split="val",
                transform=transforms.Compose([Normalizer(), Resizer()]),
            )

    print("Dataset configured")

    """
    Define the data loaders that randomly sample and feed the data to the network
    """

    # Create the training dataloader
    train_sampler = AspectRatioBasedSampler(
        train_dataset, batch_size=hp.batch, drop_last=False
    )
    train_dataloader = DataLoader(
        train_dataset, num_workers=1, collate_fn=collater, batch_sampler=train_sampler
    )

    # Create the validation dataloader
    all_val_sampler = AspectRatioBasedSampler(
        all_val_dataset, batch_size=1, drop_last=False
    )
    all_val_dataloader = DataLoader(
        all_val_dataset,
        num_workers=3,
        collate_fn=collater,
        batch_sampler=all_val_sampler,
    )

    # Create the validation dataloaders for each directory specified in dataset_names
    if multiple_val_sets:
        dataloader_val_dict = {}
        for dataset_name in hp.dataset_names:
            sampler = AspectRatioBasedSampler(
                dataset_val_dict[dataset_name], batch_size=1, drop_last=False
            )
            dataloader_val_dict[dataset_name] = DataLoader(
                dataset_val_dict[dataset_name],
                num_workers=3,
                collate_fn=collater,
                batch_sampler=sampler,
            )

    print("Dataloaders configured")

    """
    Debug section this section is used to test augmentations etc without loading the entire networks. Default off.
    """
    if debugging_flag:  # debug flag. This will end execution
        debug_augmentations(train_dataloader)

    """
    Define the model structure, load in the weights. Can initialise from coco or provided pretrained weights.
    """
    # Create the model
    if hp.init_from_coco == "True":
        # Creates retinaNet structure from model.py, only loads resnet pretrained weights onto resnet portion.
        model = retinanet.build_retina_net(
            num_classes=train_dataset.num_classes(), pretrained=True, resnet_depth=101
        )
    else:
        model = retinanet.build_retina_net(
            num_classes=train_dataset.num_classes(), pretrained=False, resnet_depth=101
        )

        state_dict = torch.load("../" + hp.pre_trained)

        # Removes the module string from the keys if it's there.
        model = load_weights(state_dict, model)

        for child in model.children():
            for param in child.parameters():
                param.requires_grad = True

    print("Model Loaded")

    """
    Option to fine tune the top classification and regression models only. (ie assume that the underlying retinanet is fine for this problem as is)
    """
    # Freeze weights of ResNet
    if hp.init_from_coco == "True":
        for child in model.children():
            for param in child.parameters():
                param.requires_grad = False
        for param in model.classificationModel.parameters():
            param.requires_grad = True
        for param in model.regressionModel.parameters():
            param.requires_grad = True

    """
    Move to GPU and define the optimiser
    """

    model = model.cuda()
    model = torch.nn.DataParallel(model).cuda()
    print("Pushed model to cuda")

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=hp.learning_rate,
        amsgrad=True,
    )
    scheduler = StepLR(optimizer, step_size=8, gamma=0.1)

    loss_hist = collections.deque(maxlen=500)

    print("Num training images: {}".format(len(train_dataset)))

    prev_best_mAP = 0
    train_stats = pd.DataFrame(columns=["epoch", "train_loss", "val_loss", "mAP_val"])

    dataloaders = {"train": train_dataloader, "val_all": all_val_dataloader}
    if multiple_val_sets:
        dataloaders.update(dataloader_val_dict)

    """
    Create directories for saving results
    """
    # Saves each run by its timestamp
    fmt = "%Y-%m-%dT%H:%M:%S"
    timestamp = datetime.strftime(datetime.utcnow(), fmt)
    run_path = Path("results") / "nuclei" / hp.exp_name / timestamp
    run_path.mkdir(parents=True, exist_ok=True)

    """
    Run training epochs!
    """
    try:
        print(
            f"Training on datasets {hp.dataset_names} for {hp.epochs} epochs, with lr of {hp.learning_rate}, "
            f"batch size {hp.batch}, and init from coco is {hp.init_from_coco}"
        )

        batch_count = 0

        for epoch_num in range(hp.epochs):

            model.train()
            epoch_losses = {}
            for phase in dataloaders.keys():
                epoch_losses[phase] = []
                for i, data in enumerate(dataloaders[phase]):
                    optimizer.zero_grad()

                    # Calculate loss
                    classification_loss, regression_loss = model(
                        [data["img"].cuda().float(), data["annot"].cuda()]
                    )
                    classification_loss = classification_loss.mean()
                    regression_loss = regression_loss.mean()
                    loss = classification_loss + regression_loss

                    # if bool(loss == 0):
                    #     continue

                    # Plot training loss at each batch iteration
                    if phase == "train":
                        if vis:
                            vis.plot(
                                "batch loss",
                                "train",
                                "Loss Per Batch",
                                "Iteration",
                                "Loss",
                                batch_count,
                                float(loss),
                            )
                        batch_count += 1

                        # Backprop model
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                        optimizer.step()

                    loss_hist.append(float(loss))
                    epoch_losses[phase].append(float(loss))

                    print(
                        f"Epoch: {epoch_num} | Phase: {phase} | Iter: {i} | "
                        f"Class loss: {float(classification_loss):1.5f} | "
                        f"Regression loss: {float(regression_loss):1.5f} | "
                        f"Running loss: {np.mean(loss_hist):1.5f}"
                    )

                    del classification_loss
                    del regression_loss
                    del loss

                # Plot losses at each epoch for training and all validation sets
                if vis:
                    vis.plot(
                        "loss",
                        phase,
                        "Loss per Epoch",
                        "Epochs",
                        "Loss",
                        epoch_num,
                        np.mean(epoch_losses[phase]),
                    )

            scheduler.step()

            # Calculate and plot mAP for all validation sets
            print("Evaluating dataset")
            mAP = evaluate(all_val_dataset, model)
            nuc_mAP = round(mAP[0][0], 4)
            print(f"all validation mAP: {nuc_mAP}")
            if vis:
                vis.plot(
                    "mAP",
                    "val_all",
                    "mAP per Epoch",
                    "Epochs",
                    "mAP",
                    epoch_num,
                    nuc_mAP,
                )
            if multiple_val_sets:
                for dataset_name in hp.dataset_names:
                    dataset_name_mAP = evaluate(dataset_val_dict[dataset_name], model)
                    dataset_mAP = round(dataset_name_mAP[0][0], 4)
                    print(f"{dataset_name} validation mAP: {dataset_mAP}")
                    if vis and dataset_name != "empty":
                        vis.plot(
                            "mAP",
                            dataset_name,
                            "mAP per Epoch",
                            "Epochs",
                            "mAP",
                            epoch_num,
                            dataset_mAP,
                        )

            # Save the best combined validation mAP model
            if prev_best_mAP == 0:
                prev_best_mAP = nuc_mAP
            elif nuc_mAP > prev_best_mAP:
                name = f"model_mAP_{nuc_mAP}.pt"
                model_weights_path = run_path / name
                torch.save(model.module.state_dict(), model_weights_path)
                prev_best_mAP = nuc_mAP

            # Save stats per epoch
            row = pd.Series(
                [
                    int(epoch_num),
                    np.mean(epoch_losses["train"]),
                    np.mean(epoch_losses["val_all"]),
                    nuc_mAP,
                ],
                index=["epoch", "train_loss", "val_loss", "mAP_val"],
            )
            train_stats = train_stats.append(row, ignore_index=True)
        train_stats.to_csv(run_path / "train_stats.csv", index=False)

        # Save parameters passed in through command line for quick reuse
        hp.to_csv(run_path)

    except KeyboardInterrupt:
        save_hp = input("Would you like to save the hyperparameters anyway? y/n: ")
        if save_hp == "y":
            hp.to_csv(run_path)


if __name__ == "__main__":
    main()
