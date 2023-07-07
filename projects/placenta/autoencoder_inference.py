import time

import typer
import torch
from torch.nn.functional import cosine_similarity, mse_loss, cosine_embedding_loss

import happy.db.eval_runs_interface as db
from happy.utils.utils import get_device, get_project_dir
from happy.organs import get_organ
from happy.utils.utils import set_seed
from happy.graph.enums import AutoEncoderModelsArg
from projects.placenta.graphs.graphs.graph_classification_utils import (
    setup_lesion_datasets,
)
from happy.graph.utils.visualise_points import visualize_points



def main(
    seed: int = 0,
    project_name: str = "placenta",
    organ_name: str = "placenta",
    exp_name: str = typer.Option(...),
    model_weights_dir: str = typer.Option(...),
    model_name: str = typer.Option(...),
    run_id: int = typer.Option(...),
    model_type: AutoEncoderModelsArg = AutoEncoderModelsArg.fps,
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
    data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
    data = data.to(device)

    # Setup trained model
    pretrained_path = (
        project_dir
        / "results"
        / "gae"
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
        out = model(data.x, data.pos, data.edge_index, data.batch)
    timer_end = time.time()
    print(f"total inference time: {timer_end - timer_start:.4f} s")

    # Evaluate over cell, tissue features or both together
    for features in ["cell", "tissue", "both"]:
        print(f"evaluating {features} features")
        if features == "cell":
            x = data.x[:, :64]
            filtered_out = out[:, :64]
        elif features == "tissue":
            x = data.x[:, 64:]
            filtered_out = out[:, 64:]
        else:
            x = data.x
            filtered_out = out

        # Compare nodewise similarity between input and output
        similarity = _get_metrics(filtered_out, x, device)

        # Plot the nodewise cosine similarity on the original graph
        save_dir = pretrained_path.parent / "eval" / f"run_{run_id}"
        save_dir.mkdir(exist_ok=True, parents=True)
        pos = data.pos.to("cpu")
        visualize_points(
            organ,
            save_dir / f"cosine_similarity_{features}.png",
            pos,
            colours=similarity,
            width=int(pos[:, 0].max()) - int(pos[:, 0].min()),
            height=int(pos[:, 1].max()) - int(pos[:, 1].min()),
        )


def _get_metrics(out, x, device):
    cosine_target = torch.ones(x.shape[0]).to(device)
    cosine_loss = cosine_embedding_loss(out, x, cosine_target).cpu()
    mse = mse_loss(out, x).cpu()
    similarity = cosine_similarity(out, x).cpu()
    print(f"cosine similarity: {similarity}")
    print(f"mean cosine similarity: {similarity.mean()}")
    print(f"cosine loss: {cosine_loss}")
    print(f"mse: {mse}")
    return similarity


if __name__ == "__main__":
    typer.run(main)
