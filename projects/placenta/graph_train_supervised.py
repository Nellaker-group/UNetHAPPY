from typing import Optional, List

import typer
import torch
from torch_geometric.transforms import ToUndirected
from torch_geometric.utils import add_self_loops
from torch_geometric.data import Batch

from happy.utils.utils import get_device
from happy.organs.organs import get_organ
from happy.logger.logger import Logger
from happy.train.utils import setup_run
from happy.utils.utils import get_project_dir
from graphs.graphs.enums import FeatureArg, MethodArg, SupervisedModelsArg
from graphs.graphs.utils import get_feature, send_graph_to_device
from graphs.graphs import graph_supervised
from graphs.graphs.create_graph import (
    get_raw_data,
    setup_graph,
    get_groundtruth_patch,
    process_knts,
)

device = get_device()


def main(
    project_name: str = "placenta",
    organ_name: str = "placenta",
    exp_name: str = typer.Option(...),
    run_ids: List[int] = typer.Option([]),
    x_min: int = 0,
    y_min: int = 0,
    width: int = -1,
    height: int = -1,
    k: int = 5,
    feature: FeatureArg = FeatureArg.embeddings,
    group_knts: bool = True,
    pretrained: Optional[str] = None,
    top_conf: bool = False,
    model_type: SupervisedModelsArg = SupervisedModelsArg.sup_graphsage,
    graph_method: MethodArg = MethodArg.k,
    batch_size: int = 51200,
    num_neighbours: int = 10,
    epochs: int = 50,
    layers: int = typer.Option(...),
    learning_rate: float = 0.001,
    weighted_loss: bool = True,
    use_custom_weights: bool = True,
    vis: bool = False,
    label_type: str = "full",
    tissue_label_tsvs: List[str] = typer.Option([]),
    val_patch_files: Optional[List[str]] = typer.Option([]),
    test_patch_files: Optional[List[str]] = typer.Option([]),
    mask_unlabelled: bool = True,
    include_validation: bool = True,
    validation_step: int = 25,
):
    model_type = model_type.value
    graph_method = graph_method.value
    feature = feature.value

    project_dir = get_project_dir(project_name)
    pretrained_path = project_dir / pretrained if pretrained else None
    organ = get_organ(organ_name)

    if len(val_patch_files) > 0:
        val_patch_files = [project_dir / "config" / file for file in val_patch_files]
    if len(test_patch_files) > 0:
        test_patch_files = [project_dir / "config" / file for file in test_patch_files]

    # Setup recording of stats per batch and epoch
    logger = Logger(
        list(["train", "train_inf", "val"]), ["loss", "accuracy"], vis=vis, file=True
    )

    datas = []
    for i, run_id in enumerate(run_ids):
        # Get training data from hdf5 files
        predictions, embeddings, coords, confidence = get_raw_data(
            project_name, run_id, x_min, y_min, width, height, top_conf
        )
        # Get ground truth manually annotated data
        _, _, tissue_class = get_groundtruth_patch(
            organ,
            project_dir,
            x_min,
            y_min,
            width,
            height,
            tissue_label_tsvs[i],
            label_type,
        )
        # Covert isolated knts into syn and turn groups into a single knt point
        if group_knts:
            predictions, embeddings, coords, confidence, tissue_class = process_knts(
                organ, predictions, embeddings, coords, confidence, tissue_class
            )
        # Covert input cell data into a graph
        feature_data = get_feature(feature, predictions, embeddings)
        data = setup_graph(coords, k, feature_data, graph_method, loop=False)
        data.y = torch.Tensor(tissue_class).type(torch.LongTensor)
        data = ToUndirected()(data)
        data.edge_index, data.edge_attr = add_self_loops(
            data["edge_index"], data["edge_attr"], fill_value="mean"
        )
        data["edge_weight"] = data["edge_attr"][:, 0]

        # Split nodes into unlabelled, training and validation sets
        if run_id == 56:
            data = graph_supervised.setup_node_splits(
                data,
                tissue_class,
                mask_unlabelled,
                include_validation,
                val_patch_files,
                test_patch_files,
            )
        else:
            if include_validation and len(val_patch_files) == 0:
                data = graph_supervised.setup_node_splits(
                    data, tissue_class, mask_unlabelled, include_validation=True
                )
            else:
                data = graph_supervised.setup_node_splits(
                    data, tissue_class, mask_unlabelled, include_validation=False
                )
        datas.append(data)

    # Combine multiple graphs into a single graph
    data = Batch.from_data_list(datas)

    # Setup the dataloader which minibatches the graph
    train_loader, val_loader = graph_supervised.setup_dataloaders(
        model_type, data, layers, batch_size, num_neighbours
    )

    # Setup the training parameters
    x, _, _ = send_graph_to_device(data, device)

    # Setup the model
    model = graph_supervised.setup_model(
        model_type, data, device, layers, len(organ.tissues), pretrained_path
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

    # Saves each run by its timestamp and record params for the run
    run_path = setup_run(project_dir, f"{model_type}/{exp_name}", "graph")
    params = graph_supervised.collect_params(
        organ_name,
        exp_name,
        run_ids,
        x_min,
        y_min,
        width,
        height,
        k,
        feature,
        graph_method,
        batch_size,
        num_neighbours,
        learning_rate,
        epochs,
        layers,
        weighted_loss,
        use_custom_weights,
    )
    params.to_csv(run_path / "params.csv", index=False)

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
                    graph_supervised.save_state(run_path, logger, model, epoch)
                    print("Saved best model")
                    prev_best_val = val_accuracy

    except KeyboardInterrupt:
        save_hp = input("Would you like to save anyway? y/n: ")
        if save_hp == "y":
            # Save the fully trained model
            graph_supervised.save_state(run_path, logger, model, epoch)

    # Save the fully trained model
    graph_supervised.save_state(run_path, logger, model, "final")
    print("Saved final model")


if __name__ == "__main__":
    typer.run(main)
