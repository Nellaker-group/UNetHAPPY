import time
from collections import defaultdict

import typer
import torch
from torch.nn.functional import (
    cosine_similarity,
    mse_loss,
    cosine_embedding_loss,
    huber_loss,
    normalize,
)
import numpy as np

import happy.db.eval_runs_interface as db
from happy.utils.utils import get_device, get_project_dir
from happy.organs import get_organ
from happy.utils.utils import set_seed
from happy.graph.enums import AutoEncoderModelsArg
from projects.placenta.graphs.graphs.graph_classification_utils import (
    setup_lesion_datasets,
)
from happy.graph.utils.visualise_points import visualize_points
from happy.models.model_builder import build_cell_classifer


def main(
    seed: int = 0,
    project_name: str = "placenta",
    organ_name: str = "placenta",
    exp_name: str = typer.Option(...),
    model_weights_dir: str = typer.Option(...),
    model_name: str = typer.Option(...),
    run_id: int = typer.Option(...),
    model_type: AutoEncoderModelsArg = AutoEncoderModelsArg.fps,
    subsample: float = 0.0,
    norm_inputs: bool = False,
    plot_similarity: bool = False,
    model_eval: bool = False,
):
    db.init()
    set_seed(seed)
    device = "cpu"
    project_dir = get_project_dir(project_name)
    organ = get_organ(organ_name)

    # get data graph and ground truth
    datasets = setup_lesion_datasets(organ, project_dir, combine=True, test=False)
    train_dataset = datasets["train"]
    val_dataset = datasets["val"]
    trainval_dataset = train_dataset.combine_with_other_dataset(val_dataset)

    data = trainval_dataset.get_data_by_run_id(run_id)
    data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
    if norm_inputs:
        data.x = normalize(data.x, p=2, dim=1)

    # Subsample data
    if subsample > 0.0:
        num_to_keep = int(data.num_nodes * subsample)
        keep_indices = np.random.choice(
            np.arange(data.num_nodes), num_to_keep, replace=False
        )
        data = data.subgraph(torch.LongTensor(keep_indices))

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
    model = torch.load(pretrained_path, map_location=device)

    # Setup save path
    save_dir = pretrained_path.parent / "eval" / f"run_{run_id}"
    save_dir.mkdir(exist_ok=True, parents=True)

    timer_start = time.time()
    # Run inference and get predicted labels for nodes
    print(f"Running inference on run {run_id}")
    model.eval()
    with torch.no_grad():
        out = model(data)
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

        pos = data.pos.cpu()
        if plot_similarity:
            # Plot the nodewise cosine similarity on the original graph
            visualize_points(
                organ,
                save_dir / f"cosine_similarity_{features}.png",
                pos,
                colours=similarity,
                width=int(pos[:, 0].max()) - int(pos[:, 0].min()),
                height=int(pos[:, 1].max()) - int(pos[:, 1].min()),
            )

        if model_eval:
            if features == "both":
                break
            # Use the final layer of the cell and tissue classifier on features
            predictions = _model_pred_on_reconstruction(
                project_dir, organ, device, features, filtered_out
            )

            if features == "cell":
                visualize_points(
                    organ,
                    save_dir / f"recon_pred_{features}.png",
                    pos,
                    labels=predictions,
                    width=int(pos[:, 0].max()) - int(pos[:, 0].min()),
                    height=int(pos[:, 1].max()) - int(pos[:, 1].min()),
                )
            else:
                colours_dict = {tissue.id: tissue.colour for tissue in organ.tissues}
                colours = [colours_dict[label] for label in predictions]
                visualize_points(
                    organ,
                    save_dir / f"recon_pred_{features}.png",
                    pos,
                    colours=colours,
                    width=int(pos[:, 0].max()) - int(pos[:, 0].min()),
                    height=int(pos[:, 1].max()) - int(pos[:, 1].min()),
                )


def _get_metrics(out, x, device):
    cosine_target = torch.ones(x.shape[0]).to(device)
    cosine_loss = cosine_embedding_loss(out, x, cosine_target).cpu()
    mse = mse_loss(out, x).cpu()
    huber = huber_loss(out, x).cpu()
    similarity = cosine_similarity(out, x).cpu()
    print(f"mean cosine similarity: {similarity.mean()}")
    print(f"cosine loss: {cosine_loss}")
    print(f"mse: {mse}")
    print(f"huber: {huber}")
    return similarity


def _model_pred_on_reconstruction(project_dir, organ, device, model_type, recon_out):
    if model_type == "cell":
        cell_model_path = (
            project_dir
            / "results"
            / "cell_class"
            / "with_towards_float_image"
            / "2023-03-11T19-41-17"
            / "cell_model_accuracy_0.871.pt"
        )
        cell_model = build_cell_classifer("resnet-50", len(organ.cells))
        cell_model.load_state_dict(
            torch.load(cell_model_path, map_location=device), strict=True
        )
        final_layer = cell_model.fc.output_layer
        predictions = final_layer(recon_out.cpu())
        predictions = torch.argmax(predictions, dim=1).numpy()
    elif model_type == "tissue":
        tissue_model_path = (
            project_dir
            / "results"
            / "graph"
            / "sup_clustergcn"
            / "reduce_dims"
            / "2023-05-23T16-46-53"
            / "600_graph_model.pt"
        )
        tissue_model = torch.load(tissue_model_path, map_location=device)
        final_layer = tissue_model.lin2
        predictions = final_layer(recon_out)
        predictions = predictions.argmax(dim=-1, keepdim=True).squeeze().cpu().numpy()
    else:
        raise ValueError(f"Unknown model type {model_type}")
    return predictions


def _rank_neighbor_similarities(x, edge_index):
    neighbor_similarities = defaultdict(list)

    for node1, node2 in edge_index:
        if node1 < node2:
            similarity = cosine_similarity(
                x[node1].unsqueeze(0), x[node2].unsqueeze(0)
            ).item()
            neighbor_similarities[node1].append((node2, similarity))
            neighbor_similarities[node2].append((node1, similarity))

    # Rank the neighbors of each node based on their similarities
    ranked_neighbors = {
        node: sorted(neighbor_similarities[node], key=lambda x: x[1], reverse=True)
        for node in neighbor_similarities
    }

    return ranked_neighbors


if __name__ == "__main__":
    typer.run(main)
