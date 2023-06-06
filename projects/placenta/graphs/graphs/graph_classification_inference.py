import time
import os

import typer
import torch
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

import happy.db.eval_runs_interface as db
from happy.utils.utils import get_device, get_project_dir
from happy.organs import get_organ
from happy.utils.utils import set_seed
from happy.graph.enums import GraphClassificationModelsArg
from projects.placenta.graphs.graphs.graph_classification_utils import (
    setup_lesion_datasets,
)


def main(
    seed: int = 0,
    project_name: str = "placenta",
    organ_name: str = "placenta",
    exp_name: str = typer.Option(...),
    model_weights_dir: str = typer.Option(...),
    model_name: str = typer.Option(...),
    run_id: int = typer.Option(...),
    model_type: GraphClassificationModelsArg = GraphClassificationModelsArg.top_k,
):
    db.init()
    set_seed(seed)
    device = get_device()
    project_dir = get_project_dir(project_name)
    organ = get_organ(organ_name)

    # get data graph and ground truth
    datasets = setup_lesion_datasets(organ, project_dir, combine=True, test=False)
    train_dataset = datasets["train"]
    val_dataset = datasets["val"]
    trainval_dataset = train_dataset.combine_with_other_dataset(val_dataset)

    data = trainval_dataset.get_data_by_run_id(run_id)
    data = data.to(device)
    lesion = torch.Tensor(trainval_dataset.get_lesion_by_run_id(run_id))

    # Setup trained model
    pretrained_path = (
        project_dir
        / "results"
        / "c_graph"
        / model_type.value
        / exp_name
        / model_weights_dir
        / model_name
    )
    model = torch.load(pretrained_path)

    timer_start = time.time()
    # Run inference and get predicted labels for nodes
    print(f"Running inference on run {run_id}")
    model.eval()
    with torch.no_grad():
        out = model(data)
    timer_end = time.time()
    print(f"total inference time: {timer_end - timer_start:.4f} s")

    # Get predicted labels for nodes
    out = out.cpu()
    pred_threshold = 0.5
    pred = ((torch.sigmoid(out)).gt(pred_threshold)).int()
    correct = (pred.eq(lesion)).float()
    accuracy = correct.mean().item()

    print(f"Ground truth: {lesion.numpy().astype(int)}")
    print(f"Predicted: {pred.numpy()}")
    print(f"Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    typer.run(main)
