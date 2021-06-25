from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from happy.train.calc_avg_precision import evaluate_ap
from happy.models import retinanet
from happy.utils.utils import load_weights
from happy.data.setup_data import setup_nuclei_datasets
from happy.data.setup_dataloader import setup_dataloaders


def setup_model(init_from_coco, device, pre_trained_path=None):
    model = retinanet.build_retina_net(
        num_classes=1, device=device, pretrained=init_from_coco, resnet_depth=101
    )
    if init_from_coco:
        for child in model.children():
            for param in child.parameters():
                param.requires_grad = False
        for param in model.classificationModel.parameters():
            param.requires_grad = True
        for param in model.regressionModel.parameters():
            param.requires_grad = True
    else:
        state_dict = torch.load(pre_trained_path)
        # Removes the module string from the keys if it's there.
        model = load_weights(state_dict, model)
        for child in model.children():
            for param in child.parameters():
                param.requires_grad = True
    model = model.to(device)
    model = torch.nn.DataParallel(model).to(device)
    print("Model Loaded")
    return model


def setup_data(annotations_path, hp, multiple_val_sets):
    datasets = setup_nuclei_datasets(
        annotations_path, hp.dataset_names, multiple_val_sets
    )
    dataloaders = setup_dataloaders(True, datasets, 3, hp.batch)
    return dataloaders


def setup_training_params(model, learning_rate):
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        amsgrad=True,
    )
    scheduler = StepLR(optimizer, step_size=8, gamma=0.1)
    return optimizer, scheduler


def setup_run(project_dir, exp_name):
    fmt = "%Y-%m-%dT%H:%M:%S"
    timestamp = datetime.strftime(datetime.utcnow(), fmt)
    run_path = project_dir / "results" / "nuclei" / exp_name / timestamp
    run_path.mkdir(parents=True, exist_ok=True)
    return run_path


def train(epochs, model, dataloaders, optimizer, logger, scheduler, run_path, device):
    prev_best_ap = 0
    batch_count = 0
    for epoch_num in range(epochs):
        model.train()
        # epoch recording metrics
        loss = {}
        for phase in dataloaders.keys():
            loss[phase] = []
            for i, data in enumerate(dataloaders[phase]):
                class_loss, regression_loss, total_loss, batch_count = single_batch(
                    phase, optimizer, model, data, logger, batch_count, device
                )
                print(
                    f"Epoch: {epoch_num} | Phase: {phase} | Iter: {i} | "
                    f"Class loss: {float(class_loss):1.5f} | "
                    f"Regression loss: {float(regression_loss):1.5f} | "
                    f"Running loss: {np.mean(logger.loss_hist):1.5f}"
                )
                # update epoch metrics
                logger.loss_hist.append(float(total_loss))
                loss[phase].append(float(total_loss))

            # Plot losses at each epoch for training and all validation sets
            logger.log_loss(phase, epoch_num, np.mean(loss[phase]))

        scheduler.step()

        # Calculate and plot mAP for all validation sets
        print("Evaluating dataset")
        prev_best_ap = validate_model(
            logger, epoch_num, prev_best_ap, model, run_path, dataloaders, device
        )


def single_batch(phase, optimizer, model, data, logger, batch_count, device):
    optimizer.zero_grad()

    # Calculate loss
    classification_loss, regression_loss = model(
        [data["img"].to(device).float(), data["annot"].to(device)], device
    )
    classification_loss = classification_loss.mean()
    regression_loss = regression_loss.mean()
    total_loss = classification_loss + regression_loss

    # Plot training loss at each batch iteration
    if phase == "train":
        logger.log_batch_loss(batch_count, float(total_loss))
        batch_count += 1
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()

    return classification_loss, regression_loss, total_loss, batch_count


def validate_model(
    logger, epoch_num, prev_best_ap, model, run_path, dataloaders, device
):
    dataloaders.pop("train")
    avg_precs = {}
    for dataset_name in dataloaders:
        dataset = dataloaders[dataset_name].dataset
        ap = evaluate_ap(dataset, model, device)
        nuc_ap = round(ap[0][0], 4)
        logger.log_ap(dataset_name, epoch_num, nuc_ap)
        avg_precs[dataset_name] = nuc_ap

    # Save the best combined validation mAP model
    if prev_best_ap != 0 and avg_precs["val_all"] > prev_best_ap:
        name = f"model_mAP_{avg_precs['val_all']}.pt"
        model_weights_path = run_path / name
        torch.save(model.module.state_dict(), model_weights_path)
        print("Model saved")

    return avg_precs["val_all"]


def save_state(logger, model, hp, run_path):
    model.eval()
    torch.save(model.module.state_dict(), run_path / "nuclei_final_model.pt")
    hp.to_csv(run_path)
    logger.train_stats.to_csv(run_path / "nuclei_train_stats.csv", index=False)
