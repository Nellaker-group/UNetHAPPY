from typing import Optional

import typer

import happy.db.eval_runs_interface as db
from happy.utils.utils import get_device
from happy.organs import get_organ
from happy.logger.logger import Logger
from happy.train.utils import setup_run
from happy.utils.utils import get_project_dir, set_seed
from happy.graph.enums import AutoEncoderModelsArg
from projects.placenta.graphs.graphs.autoencoder_runner import Params, Runner
from projects.placenta.graphs.graphs.graph_classification_utils import (
    setup_lesion_datasets,
)


def main(
    seed: int = 0,
    project_name: str = "placenta",
    organ_name: str = "placenta",
    exp_name: str = typer.Option(...),
    pretrained: Optional[str] = None,
    model_type: AutoEncoderModelsArg = AutoEncoderModelsArg.fps,
    batch_size: int = 1,
    epochs: int = 2,
    depth: int = 3,
    hidden_units: int = 128,
    pooling_ratio: float = 0.25,
    subsample_ratio: float = 0.5,
    learning_rate: float = 0.0005,
    num_workers: int = 12,
    local: bool = False,
):
    """Trains a graph classifier model on the lesion focused dataset.

    Args:
        seed: set the random seed for reproducibility
        project_name: name of project dir
        organ_name: name of organ
        exp_name: name given to this experiment for saving results and models
        pretrained: path to pretrained model (optional)
        model_type: type of model and pooling to train
        batch_size: number of graphs in each batch
        epochs: number of epochs to train for
        depth: number of pooling and unpooling layers
        hidden_units: number of hidden units in each layer
        pooling_ratio: the ratio of nodes to pool down to in each pooling layer
        subsample_ratio: the ratio of node to first subsample each graph down to
        learning_rate: the learning rate of the model
        num_workers: number of workers for the dataloader
        local: a flag to extract one local graph for testing locally
    """
    # general setup
    db.init()
    device = get_device()
    set_seed(seed)

    project_dir = get_project_dir(project_name)
    pretrained_path = project_dir / pretrained if pretrained else None
    organ = get_organ(organ_name)

    # Setup recording of stats per batch and epoch
    logger = Logger(list(["train", "val"]), ["loss", "accuracy"], vis=False, file=True)

    # Get Dataset of lesion graphs (combination of single and multi lesions)
    datasets = setup_lesion_datasets(
        organ,
        project_dir,
        combine=True,
        test=False,
        local=local
    )

    # Setup training parameters, including dataloaders, models, loss, etc.
    run_params = Params(
        datasets,
        device,
        pretrained_path,
        model_type,
        batch_size,
        epochs,
        depth,
        hidden_units,
        pooling_ratio,
        subsample_ratio,
        learning_rate,
        num_workers,
    )
    runner = Runner.new(run_params, test=False)

    # Saves each run by its timestamp and record params for the run
    run_path = setup_run(project_dir, f"{model_type.value}/{exp_name}", "gae")
    runner.params.save(seed, exp_name, run_path)

    # train!
    train(runner, logger, run_path)


def train(train_runner, logger, run_path):
    epochs = train_runner.params.epochs

    try:
        prev_best_val = 0
        for epoch in range(1, epochs + 1):
            print("Training:")
            loss, accuracy = train_runner.train()
            logger.log_loss("train", epoch - 1, loss)
            logger.log_accuracy("train", epoch - 1, accuracy)

            print("Validation:")
            val_loss, val_accuracy = train_runner.validate()
            logger.log_loss("val", epoch - 1, val_loss)
            logger.log_accuracy("val", epoch - 1, val_accuracy)
            logger.to_csv(run_path / "c_graph_train_stats.csv")
            # Save new model every epoch
            train_runner.save_state(run_path, epoch)
            if val_accuracy >= prev_best_val:
                print("Best current accuracy")
                prev_best_val = val_accuracy

    except KeyboardInterrupt:
        save_hp = input("Would you like to save anyway? y/n: ")
        if save_hp == "y":
            # Save the interrupted model
            train_runner.save_state(run_path, epoch)

    # Save the final trained model
    train_runner.save_state(run_path, "final")
    print("Saved final model")


if __name__ == "__main__":
    typer.run(main)
