from enum import Enum

import typer

from happy.utils.utils import get_device
from happy.cells.cells import get_organ
from happy.logger.logger import Logger
from happy.train.utils import setup_run
from happy.utils.utils import get_project_dir
from graphs.graphs.embeddings import generate_umap
from graphs.graphs import graph_train


class FeatureArg(str, Enum):
    predictions = "predictions"
    embeddings = "embeddings"


class MethodArg(str, Enum):
    k = "k"
    delaunay = "delaunay"


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
    top_conf: bool = False,
    graph_method: MethodArg = MethodArg.k,
    epochs: int = 50,
    layers: int = 4,
    vis: bool = True,
):
    device = get_device()
    project_dir = get_project_dir(project_name)
    organ = get_organ(organ_name)

    # Setup recording of stats per batch and epoch
    logger = Logger(list(["train"]), ["loss"], vis=vis, file=True)

    # Get data from hdf5 files
    predictions, embeddings, coords, confidence = graph_train.get_raw_data(
        project_name, run_id, x_min, y_min, width, height, top_conf
    )

    if feature.value == "predictions":
        feature_data = predictions
    elif feature.value == "embeddings":
        feature_data = embeddings
    else:
        raise ValueError(f"No such feature {feature}")

    # Create the graph from the raw data
    data = graph_train.setup_graph(coords, k, feature_data, graph_method.value)

    # Setup the dataloader which minibatches the graph
    train_loader = graph_train.setup_dataloader(data, 10240, 10)

    # Setup the model
    model, optimiser, x, edge_index = graph_train.setup_model(data, device, layers)

    # Umap plot name
    conf_str = "_top_conf" if top_conf else ""
    plot_name = f"x{x_min}_y{y_min}_w{width}_h{height}{conf_str}"

    # Saves each run by its timestamp
    run_path = setup_run(project_dir, exp_name, "graph")

    # Node embeddings before training
    generate_umap(
        model, x, edge_index, organ, predictions, run_path, f"untrained_{plot_name}"
    )

    # Train!
    print("Training:")
    for epoch in range(1, epochs + 1):
        loss = graph_train.train(model, data, x, optimiser, train_loader, device)
        logger.log_loss("train", epoch, loss)

    # Save the fully trained model
    graph_train.save_state(
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
        epochs,
        layers,
        vis,
    )


if __name__ == "__main__":
    run_id = 16
    exp_name = "delaunay_diff_layers_15000"
    x_min = 41203
    y_min = 21344
    width = 15000
    height = 15000
    graph_method = MethodArg.k
    vis = True

    for i in range(1, 3):
        epochs = 50 * i

        for j in range(1, 6):
            layers = 2 ** j

            main(
                run_id=run_id,
                exp_name=exp_name,
                x_min=x_min,
                y_min=y_min,
                width=width,
                height=height,
                epochs=epochs,
                layers=layers,
                graph_method=graph_method,
                vis=vis,
            )
