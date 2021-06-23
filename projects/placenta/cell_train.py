from datetime import datetime
from pathlib import Path
from typing import List

import typer
import numpy as np
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
from happy.cells.cells import get_organ


if torch.cuda.is_available():
    print_gpu_stats()

ImageFile.LOAD_TRUNCATED_IMAGES = True


def main(
    project_name: str = typer.Option(...),
    organ_name: str = typer.Option(...),
    exp_name: str = typer.Option(...),
    annot_dir: str = typer.Option(...),
    dataset_names: List[str] = typer.Option([]),
    model_name: str = "resnet",
    pre_trained: str = typer.Option(...),
    epochs: int = 5,
    batch: int = 200,
    learning_rate: float = 1e-5,
    init_from_coco: bool = False,
    vis: bool = True,
):
    # TODO: reimplement loading hps from file later (with database)
    hp = Hyperparameters(
        exp_name,
        annot_dir,
        dataset_names,
        model_name,
        pre_trained,
        epochs,
        batch,
        learning_rate,
        init_from_coco,
        vis,
    )
    organ = get_organ(organ_name)
    multiple_val_sets = True if len(hp.dataset_names) > 1 else False

    project_dir = Path(__file__).parent.parent.parent / "projects" / project_name
    annotations_path = project_dir / annot_dir

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

    # Get all datasets and dataloaders, including separate validation datasets
    datasets, dataloaders = setup_data(
        organ, annotations_path, hp, image_size, multiple_val_sets
    )

    logger.setup_train_stats(list(datasets.keys()), ["loss", "accuracy"])

    optimizer, criterion, scheduler = setup_training_params(model, hp.learning_rate)

    # Save each run by it's timestamp
    fmt = "%Y-%m-%dT%H:%M:%S"
    timestamp = datetime.strftime(datetime.utcnow(), fmt)
    run_path = project_dir / "results" / "cell_class" / hp.exp_name / timestamp
    run_path.mkdir(parents=True, exist_ok=True)

    # train!
    prev_best_accuracy = 0
    try:
        print(f"Num training images: {len(datasets['train'])}")
        print(
            f"Training on datasets {hp.dataset_names} for {hp.epochs} epochs, "
            f"with lr of {hp.learning_rate}, batch size {hp.batch}, "
            f"init from coco is {hp.init_from_coco}"
        )

        batch_count = 0

        for epoch_num in range(hp.epochs):
            model.train()
            # Setup recording metrics
            epoch_loss = {}
            predicted_cell = {}
            truth_cell = {}
            num_correct = {}

            for phase in dataloaders:
                print(phase)
                # Setup recording metrics
                epoch_loss[phase] = []
                predicted_cell[phase] = []
                truth_cell[phase] = []
                num_correct[phase] = 0

                if phase != "train":
                    model.eval()

                for i, data in enumerate(dataloaders[phase]):
                    batch_loss, batch_preds, batch_truth, batch_correct = single_batch(
                        phase, optimizer, criterion, model, data, logger, batch_count
                    )

                    print(
                        f"Epoch: {epoch_num} | Phase: {phase} | Iteration: {i} | "
                        f"Classification loss: {float(batch_loss):1.5f} | "
                        f"Running loss: {np.mean(logger.loss_hist):1.5f}"
                    )

                    logger.loss_hist.append(float(batch_loss))
                    epoch_loss[phase].append(float(batch_loss))
                    predicted_cell[phase].append(batch_preds)
                    truth_cell[phase].append(batch_truth)
                    num_correct[phase] += batch_correct

                # Plot losses at each epoch for training and all validation sets
                logger.log_loss(phase, epoch_num, np.mean(epoch_loss[phase]))

            scheduler.step()

            # Calculate and plot accuracy for all validation sets
            val_accuracy = evaluate_epoch(logger, num_correct, epoch_num, datasets)

            if prev_best_accuracy == 0:
                prev_best_accuracy = val_accuracy

            if val_accuracy > prev_best_accuracy:
                name = f"cell_model_accuracy_{round(val_accuracy, 4)}.pt"
                torch.save(model.module.state_dict(), run_path / name)
                print("Model saved")
                prev_best_accuracy = val_accuracy

                # Generate confusion matrix for all the validation sets
                validation_confusion_matrices(
                    logger,
                    predicted_cell,
                    truth_cell,
                    hp.dataset_names,
                    run_path,
                )

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

    save_state(model, hp, run_path)


def setup_data(organ, annotations_path, hp, image_size, multiple_val_sets):
    dataset_train, dataset_val, dataset_val_dict = _setup_datasets(
        organ, annotations_path, hp.dataset_names, image_size, multiple_val_sets
    )
    datasets = {"train": dataset_train, "val_all": dataset_val}
    if multiple_val_sets:
        datasets.update(dataset_val_dict)
    train_dataloader, all_val_dataloader, dataloader_val_dict = _setup_dataloaders(
        dataset_train, dataset_val, dataset_val_dict, hp.dataset_names, hp.batch
    )
    dataloaders = {"train": train_dataloader, "val_all": all_val_dataloader}
    if multiple_val_sets:
        dataloaders.update(dataloader_val_dict)
    return datasets, dataloaders


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


def setup_training_params(model, learning_rate):
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        amsgrad=True,
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=8, gamma=0.1)
    return optimizer, criterion, scheduler


def single_batch(phase, optimizer, criterion, model, data, logger, batch_count):
    optimizer.zero_grad()

    # Get predictions and calculate loss
    class_prediction = model(data["img"].cuda().float())
    loss = criterion(class_prediction, data["annot"].cuda())

    # Get predicted cell class and record number of correct predictions
    predicted = torch.max(class_prediction, 1)[1].cpu()
    num_correct = (predicted == data["annot"]).sum().item()

    # Plot training loss at each batch iteration
    if phase == "train":
        logger.log_batch_loss(batch_count, float(loss))
        batch_count += 1
        # Backprop model
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()

    predictions = predicted.tolist()
    ground_truths = data["annot"].tolist()
    return loss, predictions, ground_truths, num_correct


def evaluate_epoch(logger, correct, epoch_num, datasets):
    # Calculate and plot accuracy for all validation sets
    print("Evaluating datasets")
    for dataset_name in datasets:
        accuracy = int(100 * (correct[dataset_name] / len(dataset_name)))
        logger.log_accuracy(dataset_name, epoch_num, accuracy)
        if dataset_name == "all_val":
            val_accuracy = accuracy
    return val_accuracy


def validation_confusion_matrices(logger, pred, truth, dataset_names, run_path):
    # Save confusion matrix plots for all validation sets
    for dataset_name in dataset_names:
        cm = confusion_matrix(pred[dataset_name], truth[dataset_name])
        logger.log_confusion_matrix(cm, dataset_name, run_path)


def save_state(model, hp, run_path):
    model.eval()
    torch.save(model.module.state_dict(), run_path / "cell_final_model.pt")
    hp.to_csv(run_path)


def _setup_datasets(organ, annot_dir, dataset_names, image_size, multiple_val_sets):
    # TODO: change oversampled to a param rather than 'True'
    # Create the datasets from all directories specified in dataset_names
    dataset_train = get_cell_dataset(
        organ, "train", annot_dir, dataset_names, image_size, True
    )
    dataset_val = get_cell_dataset(
        organ, "val", annot_dir, dataset_names, image_size, False
    )
    # Create validation datasets from all directories specified in dataset_names
    dataset_val_dict = {}
    if multiple_val_sets:
        for dataset_name in dataset_names:
            dataset_val_dict[dataset_name] = get_cell_dataset(
                organ, "val", annot_dir, dataset_name, image_size, False
            )
    print("Dataset configured")
    return dataset_train, dataset_val, dataset_val_dict


def _setup_dataloaders(
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


if __name__ == "__main__":
    typer.run(main)
