from pathlib import Path
from typing import List

import typer
import torch

from happy.utils.hyperparameters import Hyperparameters
from happy.utils.utils import print_gpu_stats
from happy.logger.logger import Logger
from happy.cells.cells import get_organ
from happy.train import cell_train


if torch.cuda.is_available():
    print_gpu_stats()


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

    # Defines the Visdom visualisations (make sure the ports are tunneling)
    logger = Logger(hp.vis)

    # Define the model structure and load in the weights.
    # Can initialise from coco or provided pretrained weights.
    model, image_size = cell_train.setup_model(
        model_name, hp.init_from_coco, hp.pre_trained, out_features=len(organ.cells)
    )

    # Get all datasets and dataloaders, including separate validation datasets
    dataloaders = cell_train.setup_data(
        organ, project_dir / annot_dir, hp, image_size, multiple_val_sets
    )

    # Setup recording of stats per epoch
    logger.setup_train_stats(list(dataloaders.keys()), ["loss", "accuracy"])

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
            hp.epochs,
            model,
            dataloaders,
            optimizer,
            criterion,
            logger,
            scheduler,
            run_path,
        )
    except KeyboardInterrupt:
        save_hp = input("Would you like to save the hyperparameters anyway? y/n: ")
        if save_hp == "y":
            hp.to_csv(run_path)

    cell_train.save_state(logger, model, hp, run_path)


if __name__ == "__main__":
    typer.run(main)
