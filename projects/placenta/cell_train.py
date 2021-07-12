from pathlib import Path
from typing import List, Optional

import typer
import torch

from happy.utils.hyperparameters import Hyperparameters
from happy.utils.utils import set_gpu_device
from happy.logger.logger import Logger
from happy.cells.cells import get_organ
from happy.train import cell_train


if torch.cuda.is_available():
    set_gpu_device()
    device = "cuda"
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
else:
    device = "cpu"


def main(
    project_name: str = typer.Option(...),
    organ_name: str = typer.Option(...),
    exp_name: str = typer.Option(...),
    annot_dir: str = typer.Option(...),
    dataset_names: List[str] = typer.Option([]),
    model_name: str = "resnet-50",
    pre_trained: Optional[str] = None,
    num_workers: int = 12,
    epochs: int = 5,
    batch: int = 400,
    val_batch: int = 200,
    learning_rate: float = 1e-5,
    init_from_coco: bool = False,
    frozen: bool = True,
    vis: bool = True,
):
    """For training a cell classification model

    Multiple dataset can be combined by passing in 'dataset_names' multiple times with
    the correct dataset directory name.

    Visualising the batch and epoch level training stats requires having a visdom
    server running on port 8998.

    Args:
        project_name: name of the project dir to save results to
        organ_name: name of organ for getting the cells
        exp_name: name of the experiment directory to save results to
        annot_dir: relative path to annotations
        dataset_names: name of directory containing one dataset
        model_name: architecture name (currently 'resnet-50 or 'inceptionresnetv2')
        pre_trained: path to pretrained weights if starting from local weights
        num_workers: number of workers for parallel processing
        epochs: number of epochs to train for
        batch: batch size of the training set
        val_batch: batch size of the validation sets
        learning_rate: learning rate which decreases every 8 epochs
        init_from_coco: whether to use imagenet pretrained weights
        frozen: whether to freeze most of the layers. True for only fine-tuning
        vis: whether to send stats to visdom for visualisation
    """
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
        frozen,
        vis,
    )
    organ = get_organ(organ_name)
    multiple_val_sets = True if len(hp.dataset_names) > 1 else False
    project_dir = (
        Path(__file__).absolute().parent.parent.parent / "projects" / project_name
    )

    # Setup the model. Can be pretrained from coco or own weights.
    model, image_size = cell_train.setup_model(
        model_name, hp.init_from_coco, len(organ.cells), hp.pre_trained, frozen, device
    )

    # Get all datasets and dataloaders, including separate validation datasets
    dataloaders = cell_train.setup_data(
        organ,
        project_dir / annot_dir,
        hp,
        image_size,
        num_workers,
        multiple_val_sets,
        val_batch,
    )

    # Setup recording of stats per batch and epoch
    logger = Logger(hp.vis, list(dataloaders.keys()), ["loss", "accuracy"])

    # Setup training parameters
    optimizer, criterion, scheduler = cell_train.setup_training_params(
        model, hp.learning_rate
    )

    # Save each run by it's timestamp
    run_path = cell_train.setup_run(project_dir, exp_name)

    # train!
    try:
        print(f"Num training images: {len(dataloaders['train'].dataset)}")
        print(
            f"Training on datasets {hp.dataset_names} for {hp.epochs} epochs, "
            f"with lr of {hp.learning_rate}, batch size {hp.batch}, "
            f"init from coco is {hp.init_from_coco}"
        )
        cell_train.train(
            organ,
            hp.epochs,
            model,
            dataloaders,
            optimizer,
            criterion,
            logger,
            scheduler,
            run_path,
            device,
        )
    except KeyboardInterrupt:
        save_hp = input("Would you like to save the hyperparameters anyway? y/n: ")
        if save_hp == "y":
            hp.to_csv(run_path)

    # Save hyperparameters, the logged train stats, and the final model
    cell_train.save_state(logger, model, hp, run_path)


if __name__ == "__main__":
    typer.run(main)
