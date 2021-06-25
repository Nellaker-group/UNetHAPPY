from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, accuracy_score
from torch.optim.lr_scheduler import StepLR

from happy.train.utils import get_confusion_matrix
from happy.models.model_builder import build_cell_classifer
from happy.data.setup_data import setup_cell_datasets
from happy.data.setup_dataloader import setup_dataloaders


def setup_data(organ, annotations_path, hp, image_size, multiple_val_sets):
    datasets = setup_cell_datasets(
        organ, annotations_path, hp.dataset_names, image_size, multiple_val_sets
    )
    dataloaders = setup_dataloaders(False, datasets, 10, hp.batch)
    return dataloaders


def setup_model(model_name, init_from_coco, out_features, pre_trained_path, device):
    model = build_cell_classifer(model_name, out_features)
    image_size = (299, 299) if model_name == "inceptionresnetv2" else (224, 224)

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
    model = torch.nn.DataParallel(model).to(device)
    print("Model Loaded to device")
    return model, image_size


def setup_training_params(model, learning_rate):
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        amsgrad=True,
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=8, gamma=0.1)
    return optimizer, criterion, scheduler


def setup_run(project_dir, exp_name):
    fmt = "%Y-%m-%dT%H:%M:%S"
    timestamp = datetime.strftime(datetime.utcnow(), fmt)
    run_path = project_dir / "results" / "cell_class" / exp_name / timestamp
    run_path.mkdir(parents=True, exist_ok=True)
    return run_path


def train(
    epochs,
    model,
    dataloaders,
    optimizer,
    criterion,
    logger,
    scheduler,
    run_path,
    device,
):
    prev_best_accuracy = 0
    batch_count = 0
    for epoch_num in range(epochs):
        model.train()
        # epoch recording metrics
        loss = {}
        predictions = {}
        ground_truth = {}

        for phase in dataloaders:
            print(phase)
            loss[phase] = []
            predictions[phase] = []
            ground_truth[phase] = []

            if phase != "train":
                model.eval()

            for i, data in enumerate(dataloaders[phase]):
                batch_loss, batch_preds, batch_truth, batch_count = single_batch(
                    phase,
                    optimizer,
                    criterion,
                    model,
                    data,
                    logger,
                    batch_count,
                    device,
                )
                # update epoch metrics
                logger.loss_hist.append(float(batch_loss))
                loss[phase].append(float(batch_loss))
                predictions[phase].extend(batch_preds)
                ground_truth[phase].extend(batch_truth)
                print(
                    f"Epoch: {epoch_num} | Phase: {phase} | Iteration: {i} | "
                    f"Classification loss: {float(batch_loss):1.5f} | "
                    f"Running loss: {np.mean(logger.loss_hist):1.5f}"
                )

            # Plot losses at each epoch for training and all validation sets
            log_epoch_metrics(logger, epoch_num, phase, loss, predictions, ground_truth)

        scheduler.step()

        # Calculate and plot confusion matrices for all validation sets
        print("Evaluating dataset")
        prev_best_accuracy = validate_model(
            logger,
            epoch_num,
            prev_best_accuracy,
            model,
            run_path,
            predictions,
            ground_truth,
            list(dataloaders.keys()),
        )


def single_batch(phase, optimizer, criterion, model, data, logger, batch_count, device):
    optimizer.zero_grad()

    # Get predictions and calculate loss
    class_prediction = model(data["img"].to(device).float())
    loss = criterion(class_prediction, data["annot"].to(device))

    # Get predicted cell class and ground truth
    predictions = torch.max(class_prediction, 1)[1].cpu().tolist()
    ground_truths = data["annot"].tolist()

    # Plot training loss at each batch iteration
    if phase == "train":
        logger.log_batch_loss(batch_count, float(loss))
        batch_count += 1
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()

    return loss, predictions, ground_truths, batch_count


def log_epoch_metrics(logger, epoch_num, phase, loss, predictions, ground_truth):
    logger.log_loss(phase, epoch_num, np.mean(loss[phase]))
    accuracy = accuracy_score(ground_truth[phase], predictions[phase])
    logger.log_accuracy(phase, epoch_num, accuracy)


def validate_model(
    logger,
    epoch_num,
    prev_best_accuracy,
    model,
    run_path,
    predictions,
    ground_truths,
    datasets,
):
    val_accuracy = logger.train_stats.iloc[epoch_num]["val_all_accuracy"]

    if prev_best_accuracy != 0 and val_accuracy > prev_best_accuracy:
        name = f"cell_model_accuracy_{round(val_accuracy, 4)}.pt"
        torch.save(model.module.state_dict(), run_path / name)
        print("Model saved")

        # Generate confusion matrix for all the validation sets
        validation_confusion_matrices(
            logger,
            predictions,
            ground_truths,
            datasets,
            run_path,
        )
    return val_accuracy


def validation_confusion_matrices(logger, pred, truth, datasets, run_path):
    # Save confusion matrix plots for all validation sets
    datasets.remove("train")
    for dataset in datasets:
        cm = get_confusion_matrix(pred[dataset], truth[dataset])
        logger.log_confusion_matrix(cm, dataset, run_path)


def save_state(logger, model, hp, run_path):
    model.eval()
    torch.save(model.module.state_dict(), run_path / "cell_final_model.pt")
    hp.to_csv(run_path)
    logger.train_stats.to_csv(run_path / "cell_train_stats.csv", index=False)
