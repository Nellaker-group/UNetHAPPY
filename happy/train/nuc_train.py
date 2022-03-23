import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from happy.train.calc_avg_precision import evaluate_ap
from happy.train.calc_point_eval import evaluate_points_over_dataset
from happy.models import retinanet
from happy.utils.utils import load_weights
from happy.data.setup_data import setup_nuclei_datasets
from happy.data.setup_dataloader import setup_dataloaders


def setup_model(init_from_coco, device, frozen=True, pre_trained_path=None):
    model = retinanet.build_retina_net(
        num_classes=1, device=device, pretrained=init_from_coco, resnet_depth=101
    )
    if not init_from_coco:
        state_dict = torch.load(pre_trained_path, map_location=device)
        model = load_weights(state_dict, model)

    for child in model.children():
        for param in child.parameters():
            param.requires_grad = not frozen
    for param in model.classificationModel.parameters():
        param.requires_grad = True
    for param in model.regressionModel.parameters():
        param.requires_grad = True

    model = model.to(device)
    print(f"Frozen layers is {frozen}")
    print("Model loaded to device")
    return model


def setup_data(annotations_path, hp, multiple_val_sets, num_workers, val_batch):
    datasets = setup_nuclei_datasets(
        annotations_path, hp.dataset_names, multiple_val_sets
    )
    dataloaders = setup_dataloaders(True, datasets, num_workers, hp.batch, val_batch)
    return dataloaders


def setup_training_params(model, learning_rate, decay_gamma=0.1):
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        amsgrad=True,
    )
    scheduler = StepLR(optimizer, step_size=8, gamma=decay_gamma)
    return optimizer, scheduler


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
                # update epoch metrics
                logger.loss_hist.append(total_loss)
                loss[phase].append(total_loss)
                print(
                    f"Epoch: {epoch_num} | Phase: {phase} | Iter: {i} | "
                    f"Class loss: {class_loss:1.5f} | "
                    f"Regression loss: {regression_loss:1.5f} | "
                    f"Running loss: {np.mean(logger.loss_hist):1.5f}"
                )

            # Plot losses at each epoch for training and all validation sets
            logger.log_loss(phase, epoch_num, np.mean(loss[phase]))

        scheduler.step()

        # Calculate and plot AP for all validation sets
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

    return (
        float(classification_loss),
        float(regression_loss),
        float(total_loss),
        batch_count,
    )


def validate_model(
    logger, epoch_num, prev_best_ap, model, run_path, dataloaders, device
):
    val_dataloaders = dataloaders.copy()
    val_dataloaders.pop("train")

    max_detections = 500
    score_threshold = 0.4

    avg_precs = {}
    mean_precision = {}
    mean_recall = {}
    mean_f1 = {}
    num_empty_predictions = {}
    for dataset_name in val_dataloaders:
        if dataset_name != "empty":
            dataset = val_dataloaders[dataset_name].dataset
            ap = evaluate_ap(
                dataset,
                model,
                device,
                score_threshold=score_threshold,
                max_detections=max_detections,
            )
            nuc_ap = round(ap[0][0], 4)
            logger.log_ap(dataset_name, epoch_num, nuc_ap)
            avg_precs[dataset_name] = nuc_ap

        precision, recall, f1, num_empty = evaluate_points_over_dataset(
            dataloaders[dataset_name],
            model,
            device,
            score_threshold,
            max_detections,
            30,
        )
        mean_precision[dataset_name] = precision
        mean_recall[dataset_name] = recall
        mean_f1[dataset_name] = f1
        num_empty_predictions[dataset_name] = num_empty
        if dataset_name == "empty":
            logger.log_empty(dataset_name, epoch_num, num_empty)
        else:
            logger.log_precision(dataset_name, epoch_num, precision)
            logger.log_recall(dataset_name, epoch_num, recall)
            logger.log_f1(dataset_name, epoch_num, f1)

    # Save the best combined validation mAP model
    if avg_precs["val_all"] > prev_best_ap:
        name = f"model_mAP_{avg_precs['val_all']}.pt"
        model_weights_path = run_path / name
        torch.save(model.state_dict(), model_weights_path)
        print("Best model saved")

        return avg_precs["val_all"]
    else:
        return prev_best_ap


def save_state(logger, model, hp, run_path):
    model.eval()
    torch.save(model.state_dict(), run_path / "nuclei_final_model.pt")
    hp.to_csv(run_path)
    logger.to_csv(run_path / "nuclei_train_stats.csv")
