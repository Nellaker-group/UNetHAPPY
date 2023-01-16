## emil to get importing to work properly
import sys
import os

sys.path.append(os.getcwd())

from typing import Optional
from pathlib import Path

import typer

from db.models_training import Model, TrainRun
import db.eval_runs_interface as db


def main(
    path_to_model: Path = typer.Option(
        ..., exists=True, file_okay=True, dir_okay=False, resolve_path=True
    ),
    model_performance: float = typer.Option(...),
    run_name: str = typer.Option(...),
    run_type: str = typer.Option(...),
    path_to_pretrained_model: str = typer.Option(...),
    num_epochs: int = typer.Option(...),
    batch_size: int = typer.Option(...),
    init_lr: float = typer.Option(...),
    lr_step: Optional[int] = None,
    model_architecture: str = typer.Option(...),
    # emil
    database_id: int = None,
):
    """Add a trained model to the database

    Args:
        path_to_model: absolute path to final saved model
        model_performance: 0-1 value of validation performance of model
        slides_dir: absolute path to the dir containing the slides
        run_name: name of the training run which generated the model
        run_type: one of "Nuclei" or "Cell"
        path_to_pretrained_model: path from results/run_type to pretrained model
        num_epochs: number of epochs trained for
        batch_size: batch size during training
        init_lr: initial learning rate
        lr_step: epoch step at which learning rate decayed (can be None)
        model_architecture: Type of model used (e.g. retinanet)
        database_id: id of the database or .db file being written to
    """
    if database_id != None:
        db.init("Batch_"+str(database_id)+".db")
    else:
        db.init()

    train_run = TrainRun.create(
        run_name=run_name,
        type=run_type,
        pre_trained_path=path_to_pretrained_model,
        num_epochs=num_epochs,
        batch_size=batch_size,
        init_lr=init_lr,
        lr_step=lr_step,
    )

    Model.create(
        train_run=train_run,
        type=run_type,
        path=path_to_model,
        architecture=model_architecture,
        performance=model_performance,
    )


if __name__ == "__main__":
    typer.run(main)
