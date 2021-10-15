from typing import Optional

import typer

from graphs.graphs.create_graph import get_raw_data, setup_graph
from happy.utils.utils import get_device
from happy.organs.organs import get_organ
from happy.logger.logger import Logger
from happy.train.utils import setup_run
from happy.utils.utils import get_project_dir
from graphs.graphs.embeddings import get_graph_embeddings, fit_umap, plot_graph_umap
from graphs.graphs.enums import FeatureArg, MethodArg
from graphs.graphs.utils import get_feature
from graphs.graphs import graph_train

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
    model_type: str = "graphsage",
    graph_method: MethodArg = MethodArg.k,
    batch_size: int = 51200,
    num_neighbours: int = 10,
    epochs: int = 50,
    layers: int = typer.Option(...),
    learning_rate: float = 0.001,
    vis: bool = True,
    plot_pre_embeddings: bool = False,
    num_curriculum: int = 0,
    simple_curriculum: bool = False,
):
    project_dir = get_project_dir(project_name)
    pretrained_path = project_dir / pretrained if pretrained else None
    organ = get_organ(organ_name)

    # Setup recording of stats per batch and epoch
    logger = Logger(list(["train"]), ["loss"], vis=vis, file=True)

    # Get data from hdf5 files
    predictions, embeddings, coords, confidence = get_raw_data(
        project_name, run_id, x_min, y_min, width, height, top_conf
    )
    feature_data = get_feature(feature.value, predictions, embeddings)

    # Create the graph from the raw data
    data = setup_graph(coords, k, feature_data, graph_method.value)

    # Setup the dataloader which minibatches the graph
    train_loader = graph_train.setup_dataloader(
        model_type,
        data,
        layers,
        batch_size,
        num_neighbours,
        num_curriculum,
        simple_curriculum,
    )

    # Setup the model
    model = graph_train.setup_model(model_type, data, device, layers, pretrained_path)

    # Setup the training parameters
    optimiser, x, edge_index = graph_train.setup_parameters(
        data, model, learning_rate, device
    )

    # Saves each run by its timestamp
    run_path = setup_run(project_dir, f"{model_type}/{exp_name}", "graph")

    # Node embeddings before training
    if plot_pre_embeddings:
        conf_str = "_top_conf" if top_conf else ""
        plot_name = f"x{x_min}_y{y_min}_w{width}_h{height}{conf_str}"
        generate_umap(
            model,
            x,
            edge_index,
            organ,
            predictions,
            run_path,
            f"untrained_{plot_name}.png",
        )

    # Train!
    print("Training:")
    for i, epoch in enumerate(range(1, epochs + 1)):
        loss = graph_train.train(
            model_type,
            model,
            x,
            optimiser,
            batch_size,
            train_loader,
            device,
            run_path,
            epoch,
            simple_curriculum,
        )
        logger.log_loss("train", epoch, loss)
        if i % 50 == 0 and i != 0 and i != epoch:
            graph_train.save_model(model, run_path / f"{epoch}_graph_model.pt")

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
        batch_size,
        num_neighbours,
        learning_rate,
        epochs,
        layers,
        num_curriculum,
    )


def generate_umap(model, x, edge_index, organ, predictions, run_path, plot_name):
    graph_embeddings = get_graph_embeddings(model, x, edge_index)
    fitted_umap = fit_umap(graph_embeddings)
    plot_graph_umap(organ, predictions, fitted_umap, run_path, plot_name)


if __name__ == "__main__":
    typer.run(main)
