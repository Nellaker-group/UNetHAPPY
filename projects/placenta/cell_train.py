from datetime import datetime
from pathlib import Path
from typing import List

import typer
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import ImageFile
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import StepLR

from happy.data.transforms.collaters import cell_collater
from happy.utils.hyperparameters import Hyperparameters
from happy.utils.utils import print_gpu_stats
from happy.train.utils import confusion_matrix
from happy.utils.vis_plotter import VisdomLinePlotter
from happy.models.model_builder import build_cell_classifer
from happy.data.setup_data import get_cell_dataset
from happy.data.setup_dataloader import get_cell_dataloader
from happy.logger.logger import Logger


use_gpu = True  # a debug flag to allow non GPU testing of stuff. default true
if use_gpu:
    print_gpu_stats()

ImageFile.LOAD_TRUNCATED_IMAGES = True


def main(
    exp_name: str = "",
    annot_dir: str = "",
    dataset_names: List[str] = typer.Option([]),
    csv_classes: str = "",
    pre_trained: str = "",
    epochs: int = 5,
    batch: int = 200,
    learning_rate: float = 1e-5,
    init_from_coco: bool = False,
    vis: bool = True,
):
    prev_best_accuracy = 0
    model_name = "resnet"

    # TODO: reimplement loading hps from file later (with database)
    hp = Hyperparameters(
        exp_name,
        annot_dir,
        dataset_names,
        csv_classes,
        pre_trained,
        epochs,
        batch,
        learning_rate,
        init_from_coco,
        vis,
    )

    multiple_val_sets = True if len(hp.dataset_names) > 1 else False

    # Defines the Visdom visualisations (make sure the ports are tunneling)
    if hp.vis:
        vis = VisdomLinePlotter()
    logger = Logger(vis)

    # Define the model structure and load in the weights.
    # Can initialise from coco or provided pretrained weights.
    model, image_size = setup_model(
        model_name, hp.init_from_coco, hp.pre_trained, out_features=5
    )
    image_size = _get_image_size(model_name)

    # Get all datasets, including separate validation datasets
    dataset_train, dataset_val, dataset_val_dict = setup_datasets(
        hp.annot_dir, hp.dataset_names, hp.csv_classes, image_size, multiple_val_sets
    )
    datasets = {"train": dataset_train, "val_all": dataset_val}
    if multiple_val_sets:
        datasets.update(dataset_val_dict)

    train_dataloader, all_val_dataloader, dataloader_val_dict = setup_dataloaders(
        dataset_train, dataset_val, dataset_val_dict, hp.dataset_names, hp.batch
    )
    dataloaders = {"train": train_dataloader, "val_all": all_val_dataloader}
    if multiple_val_sets:
        dataloaders.update(dataloader_val_dict)

    logger.setup_train_stats(list(datasets.keys()), ["loss", "accuracy"])

    optimizer, criterion, scheduler = setup_training_params(model, hp.learning_rate)

    # Save each run by it's timestamp
    fmt = "%Y-%m-%dT%H:%M:%S"
    timestamp = datetime.strftime(datetime.utcnow(), fmt)
    run_path = Path("results") / "cell_class" / hp.exp_name / timestamp
    run_path.mkdir(parents=True, exist_ok=True)

    # train!
    try:
        print(f"Num training images: {len(dataset_train)}")
        print(
            f"Training on datasets {hp.dataset_names} for {hp.epochs} epochs, "
            f"with lr of {hp.learning_rate}, batch size {hp.batch}, "
            f"init from coco is {hp.init_from_coco}"
        )

        batch_count = 0

        for epoch_num in range(hp.epochs):
            model.train()
            # Setup recording metrics
            epoch_losses = {}
            truth_cell = {}
            predicted_cell = {}
            correct = {}

            for phase in dataloaders.keys():
                print(phase)
                # Setup recording metrics
                epoch_losses[phase] = []
                truth_cell[phase] = []
                predicted_cell[phase] = []
                correct[phase] = 0

                if phase != "train":
                    model.eval()

                for i, data in enumerate(dataloaders[phase]):
                    optimizer.zero_grad()

                    # Get predictions and calculate loss
                    class_prediction = model(data["img"].cuda().float())
                    loss = criterion(class_prediction, data["annot"].cuda())

                    # Returns predicted cell class and records how many in the batch were correct
                    predicted = torch.max(class_prediction, 1)[1].cpu()
                    correct[phase] += (predicted == data["annot"]).sum().item()

                    # Get accuracy values for plotting and confusion matrix of validation sets
                    if phase != "train":
                        predicted_cell[phase].append(predicted.tolist())
                        truth_cell[phase].append(data["annot"].tolist())

                    # Plot training loss at each batch iteration
                    if phase == "train":
                        logger.log_batch_loss(batch_count, float(loss))
                        batch_count += 1

                        # Backprop model
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                        optimizer.step()

                    logger.loss_hist.append(float(loss))
                    epoch_losses[phase].append(float(loss))

                    print(
                        f"Epoch: {epoch_num} | Phase: {phase} | Iteration: {i} | "
                        f"Classification loss: {float(loss):1.5f} | "
                        f"Running loss: {np.mean(logger.loss_hist):1.5f}"
                    )

                    # TODO: remove these by turning this into a function
                    # Remove tensors from GPU to avoid CUDA memory problems
                    del loss
                    del predicted
                    del class_prediction

                # Plot losses at each epoch for training and all validation sets
                logger.log_loss(phase, epoch_num, np.mean(epoch_losses[phase]))

            scheduler.step()

            # Calculate and plot accuracy for all validation sets
            val_accuracy = evaluate_epoch(logger, correct, epoch_num, datasets)

            if prev_best_accuracy == 0:
                prev_best_accuracy = val_accuracy
            elif val_accuracy > prev_best_accuracy:
                name = f"cell_model_accuracy_{round(val_accuracy, 4)}.pt"
                torch.save(model.module.state_dict(), run_path / name)
                print("Model saved")
                prev_best_accuracy = val_accuracy

                # Generate confusion matrix for the combined validation set
                all_val_cm = confusion_matrix(
                    predicted_cell["val_all"], truth_cell["val_all"]
                )
                logger.log_confusion_matrix(all_val_cm, "val_all", run_path)

                # Save confusion matrix plots for all validation sets
                if multiple_val_sets:
                    for dataset_name in hp.dataset_names:
                        cm = confusion_matrix(
                            predicted_cell[dataset_name], truth_cell[dataset_name]
                        )
                        logger.log_confusion_matrix(cm, dataset_name, run_path)

        #     # Save stats per epoch
        #     row = pd.Series(
        #         [
        #             epoch_num,
        #             np.mean(epoch_losses["train"]),
        #             np.mean(epoch_losses["val_all"]),
        #             # train_accuracy,
        #             val_accuracy,
        #         ],
        #         index=[
        #             "epoch",
        #             "train_loss",
        #             "val_loss",
        #             # "train_accuracy",
        #             "val_accuracy",
        #         ],
        #     )
        #     train_stats = train_stats.append(row, ignore_index=True)
        # logger.train_stats.to_csv(run_path / "cell_train_stats.csv", index=False)

    except KeyboardInterrupt:
        save_hp = input("Would you like to save the hyperparameters anyway? y/n: ")
        if save_hp == "y":
            hp.to_csv(run_path)

    model.eval()
    torch.save(model.module.state_dict(), run_path / "cell_final_model.pt")
    hp.to_csv(run_path)


def setup_model(model_name, init_from_coco, pre_trained_path, out_features):
    model = build_cell_classifer(model_name, out_features)

    if not init_from_coco:
        model.load_state_dict(torch.load(pre_trained_path), strict=True)
        for child in model.children():
            for param in child.parameters():
                param.requires_grad = True
    else:
        # Freeze weights of everything except classification layer
        print("Fine tuning classifier layer only. All else frozen")
        for child in model.children():
            for param in child.parameters():
                param.requires_grad = False
        if model_name == "inceptionresnetv2":
            for param in model.last_linear.parameters():
                param.requires_grad = True
        else:
            for param in model.fc.parameters():
                param.requires_grad = True

    # Move to GPU and define the optimiser
    model = torch.nn.DataParallel(model).cuda()
    print("Model Loaded to cuda")
    return model


def _get_image_size(model_name):
    if model_name == "inceptionresnetv2":
        return (299, 299)
    else:
        return (224, 224)


def setup_datasets(
    annot_dir, dataset_names, csv_classes, image_size, multiple_val_sets
):
    # TODO: change oversampled to a param rather than 'True'
    # Create the datasets from all directories specified in dataset_names
    dataset_train = get_cell_dataset(
        "train", annot_dir, dataset_names, csv_classes, image_size, True
    )
    dataset_val = get_cell_dataset(
        "val", annot_dir, dataset_names, csv_classes, image_size, False
    )
    # Create validation datasets from all directories specified in dataset_names
    dataset_val_dict = {}
    if multiple_val_sets:
        for dataset_name in dataset_names:
            dataset_val_dict[dataset_name] = get_cell_dataset(
                "val", annot_dir, dataset_name, csv_classes, image_size, False
            )
    print("Dataset configured")
    return dataset_train, dataset_val, dataset_val_dict


def setup_dataloaders(
    dataset_train, dataset_val, dataset_val_dict, dataset_names, batch_size
):
    train_dataloader = get_cell_dataloader(
        "train", dataset_train, cell_collater, batch_size
    )
    all_val_dataloader = get_cell_dataloader("val", dataset_val, cell_collater)
    # Create the validation dataloaders for each directory specified in dataset_names
    dataloader_val_dict = {}
    if dataset_val_dict:
        for dataset_name in dataset_names:
            dataloader_val_dict[dataset_name] = get_cell_dataloader(
                "val", dataset_val_dict[dataset_name], cell_collater
            )
    print("Dataloaders configured")
    return train_dataloader, all_val_dataloader, dataloader_val_dict


def setup_training_params(model, learning_rate):
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        amsgrad=True,
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=8, gamma=0.1)
    return optimizer, criterion, scheduler


def evaluate_epoch(logger, correct, epoch_num, datasets):
    # Calculate and plot accuracy for all validation sets
    val_accuracy = 0
    print("Evaluating datasets")
    for dataset_name in datasets:
        accuracy = int(100 * (correct[dataset_name] / len(dataset_name)))
        logger.log_accuracy(dataset_name, epoch_num, accuracy)
        if dataset_name == "all_val":
            val_accuracy = accuracy
    return val_accuracy



if __name__ == "__main__":
    typer.run(main)
