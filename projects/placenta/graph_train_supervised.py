from typing import Optional

import typer
import torch
from torch_geometric.transforms import ToUndirected

from graphs.graphs.create_graph import get_raw_data, setup_graph, get_groundtruth_patch
from happy.utils.utils import get_device
from happy.organs.organs import get_organ
from happy.logger.logger import Logger
from happy.train.utils import setup_run
from happy.utils.utils import get_project_dir
from graphs.graphs.enums import FeatureArg, MethodArg
from graphs.graphs.utils import get_feature, send_graph_to_device, save_model
from graphs.graphs import graph_supervised

device = get_device()


def main(
    project_name: str = "placenta",
    organ_name: str = "placenta",
    exp_name: str = typer.Option(...),
    run_id: int = typer.Option(...),
    x_min: int = 0,
    y_min: int = 0,
    width: int = -1,
    height: int = -1,
    k: int = 6,
    feature: FeatureArg = FeatureArg.embeddings,
    pretrained: Optional[str] = None,
    top_conf: bool = False,
    model_type: str = "sup_graphsage",
    graph_method: MethodArg = MethodArg.k,
    batch_size: int = 51200,
    num_neighbours: int = 10,
    epochs: int = 50,
    layers: int = typer.Option(...),
    learning_rate: float = 0.001,
    vis: bool = True,
    label_type: str = "full",
    tissue_label_tsv: str = "139_tissue_points.tsv",
):
    project_dir = get_project_dir(project_name)
    pretrained_path = project_dir / pretrained if pretrained else None
    organ = get_organ(organ_name)

    # Setup recording of stats per batch and epoch
    logger = Logger(list(["train"]), ["loss", "accuracy"], vis=vis, file=True)

    # Get data from hdf5 files
    predictions, embeddings, coords, confidence = get_raw_data(
        project_name, run_id, x_min, y_min, width, height, top_conf
    )
    feature_data = get_feature(feature.value, predictions, embeddings)
    _, _, tissue_class = get_groundtruth_patch(
        organ, project_dir, x_min, y_min, width, height, tissue_label_tsv, label_type
    )
    num_classes = len(organ.tissues)

    # Create the graph from the raw data
    data = setup_graph(coords, k, feature_data, graph_method.value)
    data.y = torch.Tensor(tissue_class).type(torch.LongTensor)
    if model_type == "sup_clustergcn":
        data = ToUndirected()(data)

    # Setup the dataloader which minibatches the graph
    train_loader = graph_supervised.setup_dataloader(
        model_type, data, layers, batch_size, num_neighbours
    )

    # Setup the training parameters
    x, edge_index, tissue_class = send_graph_to_device(data, device, tissue_class)

    # Setup the model
    model = graph_supervised.setup_model(
        model_type, data, device, layers, pretrained_path, num_classes
    )
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Saves each run by its timestamp
    run_path = setup_run(project_dir, f"{model_type}/{exp_name}", "graph")

    # Train!
    try:
        print("Training:")
        for epoch in range(1, epochs + 1):
            loss, accuracy = graph_supervised.train(
                model_type,
                model,
                x,
                optimiser,
                train_loader,
                device,
                tissue_class,
            )
            logger.log_loss("train", epoch - 1, loss)
            logger.log_accuracy("train", epoch - 1, accuracy)

            if epoch % 50 == 0 and epoch != epochs:
                save_model(model, run_path / f"{epoch}_graph_model.pt")
    except KeyboardInterrupt:
        save_hp = input("Would you like to save the hyperparameters anyway? y/n: ")
        if save_hp == "y":
            # Save the fully trained model
            graph_supervised.save_state(
                run_path,
                logger,
                model,
                organ_name,
                exp_name,
                run_id,
                x_min,
                y_min,
                width,
                height,
                k,
                feature,
                top_conf,
                graph_method,
                batch_size,
                num_neighbours,
                learning_rate,
                epochs,
                layers,
                label_type,
            )

    # Save the fully trained model
    graph_supervised.save_state(
        run_path,
        logger,
        model,
        organ_name,
        exp_name,
        run_id,
        x_min,
        y_min,
        width,
        height,
        k,
        feature,
        top_conf,
        graph_method,
        batch_size,
        num_neighbours,
        learning_rate,
        epochs,
        layers,
        label_type,
    )


if __name__ == "__main__":
    typer.run(main)
