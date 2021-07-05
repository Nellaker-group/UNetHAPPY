from pathlib import Path
from typing import List, Optional

import typer
import torch

from happy.utils.hyperparameters import Hyperparameters
from happy.utils.utils import set_gpu_device
from happy.logger.logger import Logger
from happy.train import nuc_train


if torch.cuda.is_available():
    set_gpu_device()
    device = "cuda"
else:
    device = "cpu"


def main(
    project_name: str = typer.Option(...),
    exp_name: str = typer.Option(...),
    annot_dir: str = typer.Option(...),
    dataset_names: List[str] = typer.Option([]),
    model_name: str = "retinanet",
    pre_trained: Optional[str] = None,
    epochs: int = 5,
    batch: int = 5,
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
    multiple_val_sets = True if len(hp.dataset_names) > 1 else False
    project_dir = Path(__file__).parent.parent.parent / "projects" / project_name

    # Defines the Visdom visualisations (make sure the ports are tunneling)
    logger = Logger(hp.vis)

    # Setup the model. Can be pretrained from coco or own weights.
    model = nuc_train.setup_model(hp.init_from_coco, device, pre_trained)

    # Get all datasets and dataloaders, including separate validation datasets
    dataloaders = nuc_train.setup_data(project_dir / annot_dir, hp, multiple_val_sets)

    # Setup training parameters
    optimizer, scheduler = nuc_train.setup_training_params(model, hp.learning_rate)

    # Setup recording of stats per epoch
    logger.setup_train_stats(list(dataloaders.keys()), ["loss", "AP"])

    # Saves each run by its timestamp
    run_path = nuc_train.setup_run(project_dir, exp_name)

    # train!
    try:
        print(f"Num training images: {len(dataloaders['train'].dataset)}")
        print(
            f"Training on datasets {hp.dataset_names} for {hp.epochs} epochs, "
            f"with lr of {hp.learning_rate}, batch size {hp.batch}, and init from coco "
            f"is {hp.init_from_coco}"
        )
        nuc_train.train(
            epochs, model, dataloaders, optimizer, logger, scheduler, run_path, device
        )
    except KeyboardInterrupt:
        save_hp = input("Would you like to save the hyperparameters anyway? y/n: ")
        if save_hp == "y":
            hp.to_csv(run_path)

    # Save hyperparameters, the logged train stats, and the final model
    nuc_train.save_state(logger, model, hp, run_path)


if __name__ == "__main__":
    typer.run(main)
