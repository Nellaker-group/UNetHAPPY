import copy

import torch
from torch_geometric.loader import (
    ClusterData,
    ClusterLoader,
    NeighborSampler,
    NeighborLoader,
    DataLoader,
)
from torch_geometric.transforms import RandomNodeSplit
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd

from happy.models.graphsage import SupervisedSAGE, SupervisedDiffPool
from happy.models.clustergcn import ClusterGCN, ClusterGCNConvNet
from happy.models.gat import GAT, GATv2
from projects.placenta.graphs.graphs.create_graph import get_nodes_within_tiles


def setup_node_splits(
    data, tissue_class, mask_unlabelled, include_validation, val_patch_coords
):
    # Mark everything as training data first
    train_mask = torch.ones(data.num_nodes, dtype=torch.bool)
    data.train_mask = train_mask

    # Mask unlabelled data to ignore during training
    if mask_unlabelled:
        unlabelled_inds = (tissue_class == 0).nonzero()[0]
        unlabelled_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        unlabelled_mask[unlabelled_inds] = True
        data.unlabelled_mask = unlabelled_mask
        train_mask[unlabelled_inds] = False
        data.train_mask = train_mask
        print(f"{len(unlabelled_inds)} nodes marked as unlabelled")

    # Split the graph by masks into training and validation nodes
    if include_validation:
        if val_patch_coords[0] is None and not mask_unlabelled:
            print("No validation patched provided, splitting nodes randomly")
            data = RandomNodeSplit(num_val=0.3, num_test=0.0)(data)
        else:
            print("Splitting graph by validation patch")
            val_node_inds = get_nodes_within_tiles(
                (val_patch_coords[0], val_patch_coords[1]),
                val_patch_coords[2],
                val_patch_coords[3],
                data["pos"][:, 0],
                data["pos"][:, 1],
            )
            val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            val_mask[val_node_inds] = True
            data.val_mask = val_mask
            train_mask[val_node_inds] = False
            data.train_mask = train_mask
        print(
            f"Graph split into {data.train_mask.sum().item()} train nodes "
            f"and {data.val_mask.sum().item()} validation nodes"
        )
    return data


def setup_dataloaders(
    model_type,
    data,
    num_layers,
    batch_size,
    num_neighbors,
):
    if model_type == "sup_clustergcn" or model_type == "sup_gat":
        cluster_data = ClusterData(
            data, num_parts=int(data.x.size()[0] / num_neighbors), recursive=False
        )
        train_loader = ClusterLoader(
            cluster_data, batch_size=batch_size, shuffle=True, num_workers=12
        )
        val_loader = NeighborSampler(
            data.edge_index, sizes=[-1], batch_size=1024, shuffle=False, num_workers=12
        )
    else:
        train_loader = NeighborLoader(
            data,
            input_nodes=data.train_mask,
            num_neighbors=[num_neighbors for _ in range(num_layers)],
            batch_size=batch_size,
            shuffle=True,
            num_workers=12,
        )
        val_loader = NeighborLoader(
            copy.copy(data), num_neighbors=[-1], shuffle=False, batch_size=512
        )
        val_loader.data.num_nodes = data.num_nodes
        val_loader.data.n_id = torch.arange(data.num_nodes)

    return train_loader, val_loader


def setup_model(model_type, data, device, layers, num_classes, pretrained=None):
    if pretrained:
        return torch.load(pretrained / "graph_model.pt", map_location=device)
    if model_type == "sup_graphsage":
        model = SupervisedSAGE(
            data.num_node_features,
            hidden_channels=64,
            out_channels=num_classes,
            num_layers=layers,
        )
    elif model_type == "sup_gat":
        model = GAT(
            data.num_node_features,
            hidden_channels=128,
            out_channels=num_classes,
            heads=1,
            num_layers=layers,
        )
    elif model_type == "sup_diffpool":
        model = SupervisedDiffPool(
            data.num_node_features,
            hidden_channels=64,
            out_channels=num_classes,
            num_sage_layers=layers,
        )
    elif model_type == "sup_clustergcn":
        model = ClusterGCN(
            data.num_node_features,
            hidden_channels=400,
            out_channels=num_classes,
            num_layers=layers,
        )
    else:
        return ValueError(f"No such model type implemented: {model_type}")
    model = model.to(device)
    return model


def setup_training_params(
    model,
    model_type,
    organ,
    learning_rate,
    train_dataloader,
    device,
    weighted_loss,
    use_custom_weights,
):
    if model_type == "sup_graphsage":
        if weighted_loss:
            data_classes = train_dataloader.data.y[
                train_dataloader.data.train_mask
            ].numpy()
            class_weights = _compute_tissue_weights(
                data_classes, organ, use_custom_weights
            )
            class_weights = torch.FloatTensor(class_weights)
            class_weights = class_weights.to(device)
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = torch.nn.CrossEntropyLoss()
    elif model_type == "sup_clustergcn" or model_type == "sup_gat":
        if weighted_loss:
            data_classes = train_dataloader.cluster_data.data.y[
                train_dataloader.cluster_data.data.train_mask
            ].numpy()
            class_weights = _compute_tissue_weights(
                data_classes, organ, use_custom_weights
            )
            class_weights = torch.FloatTensor(class_weights)
            class_weights = class_weights.to(device)
            criterion = torch.nn.NLLLoss(weight=class_weights)
        else:
            criterion = torch.nn.NLLLoss()
    else:
        raise ValueError(f"No such model type {model_type}")
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return optimiser, criterion


def train(
    model_type,
    model,
    optimiser,
    criterion,
    train_loader,
    device,
):
    model.train()

    total_loss = 0
    total_examples = 0
    total_correct = 0

    if model_type == "sup_clustergcn" or model_type == "sup_gat":
        for batch in train_loader:
            batch = batch.to(device)
            optimiser.zero_grad()
            out = model(batch.x, batch.edge_index)
            train_out = out[batch.train_mask]
            train_y = batch.y[batch.train_mask]
            loss = criterion(train_out, train_y)
            loss.backward()
            optimiser.step()

            nodes = batch.train_mask.sum().item()
            total_loss += loss.item() * nodes
            total_correct += int(train_out.argmax(dim=-1).eq(train_y).sum().item())
            total_examples += nodes
    else:
        for batch in train_loader:
            batch = batch.to(device)
            optimiser.zero_grad()
            train_y = batch.y[: batch.batch_size][batch.train_mask[: batch.batch_size]]
            out = model(batch.x, batch.edge_index)[: batch.batch_size]
            train_out = out[batch.train_mask[: batch.batch_size]]
            loss = criterion(train_out, train_y)
            loss.backward()
            optimiser.step()

            nodes = batch.train_mask[: batch.batch_size].sum().item()
            total_loss += float(loss) * nodes
            total_correct += int((train_out.argmax(dim=-1).eq(train_y)).sum())
            total_examples += nodes

    return total_loss / total_examples, total_correct / total_examples


@torch.no_grad()
def validate(model, data, eval_loader, device):
    model.eval()
    out, _ = model.inference(data.x, eval_loader, device)
    out = out.argmax(dim=-1)
    y = data.y.to(out.device)
    train_accuracy = int((out[data.train_mask].eq(y[data.train_mask])).sum()) / int(
        data.train_mask.sum()
    )
    val_accuracy = int((out[data.val_mask].eq(y[data.val_mask])).sum()) / int(
        data.val_mask.sum()
    )
    return train_accuracy, val_accuracy


@torch.no_grad()
def inference(model, x, eval_loader, device):
    print("Running inference")
    model.eval()
    out, graph_embeddings = model.inference(x, eval_loader, device)
    predicted_labels = out.argmax(dim=-1, keepdim=True).squeeze()
    predicted_labels = predicted_labels.cpu().numpy()
    out = out.cpu().detach().numpy()
    return out, graph_embeddings, predicted_labels


def save_state(
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
):
    torch.save(model, run_path / "graph_model.pt")

    params_df = pd.DataFrame(
        {
            "organ_name": organ_name,
            "exp_name": exp_name,
            "run_id": run_id,
            "x_min": x_min,
            "y_min": y_min,
            "width": width,
            "height": height,
            "k": k,
            "feature": feature.value,
            "top_conf": top_conf,
            "graph_method": graph_method.value,
            "batch_size": batch_size,
            "num_neighbours": num_neighbours,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "layers": layers,
            "label_type": label_type,
        },
        index=[0],
    )
    params_df.to_csv(run_path / "params.csv", index=False)

    logger.to_csv(run_path / "graph_train_stats.csv")


def _compute_tissue_weights(data_classes, organ, use_custom_weights):
    unique_classes = np.unique(data_classes)
    if not use_custom_weights:
        weighting = "balanced"
    else:
        custom_weights = [1, 0.67, 0.9, 10.5, 0.8, 1.3, 5.6, 3, 77]
        weighting = dict(zip(list(unique_classes), custom_weights))
    class_weights = compute_class_weight(
        weighting, classes=unique_classes, y=data_classes
    )
    # Account for missing tissues in training data
    classes_in_training = set(unique_classes)
    all_classes = {tissue.id for tissue in organ.tissues}
    missing_classes = list(all_classes - classes_in_training)
    missing_classes.sort()
    for i in missing_classes:
        class_weights = np.insert(class_weights, i, 0.0)
    return class_weights
