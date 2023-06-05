from typing import Optional

import typer

import happy.db.eval_runs_interface as db
from happy.utils.utils import get_device
from happy.organs import get_organ
from happy.logger.logger import Logger
from happy.train.utils import setup_run
from happy.utils.utils import get_project_dir, set_seed
from happy.graph.enums import GraphClassificationModelsArg
from projects.placenta.graphs.graphs.graph_classifier_runner import Params, Runner
from projects.placenta.graphs.graphs.lesion_dataset import LesionDataset


def main(
    seed: int = 0,
    project_name: str = "placenta",
    organ_name: str = "placenta",
    exp_name: str = typer.Option(...),
    pretrained: Optional[str] = None,
    model_type: GraphClassificationModelsArg = GraphClassificationModelsArg.top_k,
    batch_size: int = 3,
    epochs: int = 100,
    layers: int = 3,
    hidden_units: int = 128,
    pooling_ratio: float = 0.8,
    learning_rate: float = 0.001,
    num_workers: int = 12,
    subsample_ratio: float = 0.0,
):
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
    datasets = setup_lesion_datasets(organ, project_dir, combine=True, test=False)

    # Setup training parameters, including dataloaders, models, loss, etc.
    run_params = Params(
        datasets,
        device,
        pretrained_path,
        model_type,
        batch_size,
        epochs,
        layers,
        hidden_units,
        pooling_ratio,
        learning_rate,
        num_workers,
        subsample_ratio,
        organ,
    )
    runner = Runner.new(run_params, test=False)

    # Saves each run by its timestamp and record params for the run
    run_path = setup_run(project_dir, f"{model_type.value}/{exp_name}", "c_graph")
    runner.params.save(seed, exp_name, run_path)

    # train!
    train(runner, logger, run_path)


def train(train_runner, logger, run_path):
    epochs = train_runner.params.epochs

    try:
        print("Training:")
        prev_best_val = 0
        for epoch in range(1, epochs + 1):
            loss, accuracy = train_runner.train()
            logger.log_loss("train", epoch - 1, loss)
            logger.log_accuracy("train", epoch - 1, accuracy)

            val_loss, val_accuracy = train_runner.validate()
            logger.log_loss("train", epoch - 1, val_loss)
            logger.log_accuracy("val", epoch - 1, val_accuracy)
            logger.to_csv(run_path / "c_graph_train_stats.csv")
            # Save new best model
            if val_accuracy >= prev_best_val:
                train_runner.save_state(run_path, epoch)
                print("Saved best model")
                prev_best_val = val_accuracy

    except KeyboardInterrupt:
        save_hp = input("Would you like to save anyway? y/n: ")
        if save_hp == "y":
            # Save the interrupted model
            train_runner.save_state(run_path, epoch)

    # Save the final trained model
    train_runner.save_state(run_path, "final")
    print("Saved final model")


def setup_lesion_datasets(organ, project_dir, combine=True, test=False):
    datasets = {}
    if combine:
        single_lesion_train_data = LesionDataset(organ, project_dir, "single", "train")
        multi_lesion_train_data = LesionDataset(organ, project_dir, "multi", "train")
        datasets["train"] = single_lesion_train_data.combine_with_other_dataset(
            multi_lesion_train_data
        )
        single_lesion_val_data = LesionDataset(organ, project_dir, "single", "val")
        multi_lesion_val_data = LesionDataset(organ, project_dir, "multi", "val")
        datasets["val"] = single_lesion_val_data.combine_with_other_dataset(
            multi_lesion_val_data
        )
        if test:
            single_lesion_test_data = LesionDataset(
                organ, project_dir, "single", "test"
            )
            multi_lesion_test_data = LesionDataset(organ, project_dir, "multi", "test")
            datasets["test"] = single_lesion_test_data.combine_with_other_dataset(
                multi_lesion_test_data
            )
    else:
        datasets["train_single"] = LesionDataset(organ, project_dir, "single", "train")
        datasets["train_multi"] = LesionDataset(organ, project_dir, "multi", "train")
        datasets["val_single"] = LesionDataset(organ, project_dir, "single", "val")
        datasets["val_multi"] = LesionDataset(organ, project_dir, "multi", "val")
        if test:
            datasets["test_single"] = LesionDataset(
                organ, project_dir, "single", "test"
            )
            datasets["test_multi"] = LesionDataset(organ, project_dir, "multi", "test")
    return datasets


if __name__ == "__main__":
    typer.run(main)
