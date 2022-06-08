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
    weighted_loss: bool = False,
    use_custom_weights: bool = True,
    vis: bool = True,
    label_type: str = "full",
    tissue_label_tsv: str = "139_tissue_points.tsv",
    val_x_min: Optional[int] = None,
    val_y_min: Optional[int] = None,
    val_width: Optional[int] = None,
    val_height: Optional[int] = None,
    mask_unlabelled: bool = True,
    include_validation: bool = True,
    validation_step: int = 25,
):
    project_dir = get_project_dir(project_name)
    pretrained_path = project_dir / pretrained if pretrained else None
    organ = get_organ(organ_name)

    # Setup recording of stats per batch and epoch
    logger = Logger(
        list(["train", "train_inf", "val"]), ["loss", "accuracy"], vis=vis, file=True
    )

    # Get training data from hdf5 files
    predictions, embeddings, coords, confidence = get_raw_data(
        project_name, run_id, x_min, y_min, width, height, top_conf
    )
    feature_data = get_feature(feature.value, predictions, embeddings)
    _, _, tissue_class = get_groundtruth_patch(
        organ, project_dir, x_min, y_min, width, height, tissue_label_tsv, label_type
    )
    num_classes = len(organ.tissues)

    # Create the graph from the raw data
    data = setup_graph(coords, k, feature_data, graph_method.value, loop=False)
    data.y = torch.Tensor(tissue_class).type(torch.LongTensor)
    if model_type == "sup_clustergcn":
        data = ToUndirected()(data)

    # Split nodes into unlabelled, training and validation sets
    val_patch_coords = (val_x_min, val_y_min, val_width, val_height)
    data = graph_supervised.setup_node_splits(
        data, tissue_class, mask_unlabelled, include_validation, val_patch_coords
    )

    # Setup the dataloader which minibatches the graph
    train_loader, val_loader = graph_supervised.setup_dataloaders(
        model_type, data, layers, batch_size, num_neighbours
    )

    # Setup the training parameters
    x, _, _ = send_graph_to_device(data, device)

    # Setup the model
    model = graph_supervised.setup_model(
        model_type, data, device, layers, pretrained_path, num_classes
    )

    # Setup training parameters
    optimiser, criterion = graph_supervised.setup_training_params(
        model,
        model_type,
        organ,
        learning_rate,
        train_loader,
        device,
        weighted_loss,
        use_custom_weights,
    )

    # Saves each run by its timestamp
    run_path = setup_run(project_dir, f"{model_type}/{exp_name}", "graph")

    # Train!
    try:
        print("Training:")
        prev_best_val = 0
        for epoch in range(1, epochs + 1):
            loss, accuracy = graph_supervised.train(
                model_type,
                model,
                optimiser,
                criterion,
                train_loader,
                device,
            )
            logger.log_loss("train", epoch - 1, loss)
            logger.log_accuracy("train", epoch - 1, accuracy)

            if include_validation and (epoch % validation_step == 0 or epoch == 1):
                train_accuracy, val_accuracy = graph_supervised.validate(
                    model, data, val_loader, device
                )
                logger.log_accuracy("train_inf", epoch - 1, train_accuracy)
                logger.log_accuracy("val", epoch - 1, val_accuracy)

                # Save new best model
                if val_accuracy >= prev_best_val:
                    save_model(model, run_path / f"{epoch}_graph_model.pt")
                    print("Saved best model")
                    prev_best_val = val_accuracy

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
