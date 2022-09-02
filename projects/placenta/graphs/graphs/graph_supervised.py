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
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    top_k_accuracy_score,
    f1_score,
    cohen_kappa_score,
    roc_auc_score,
    matthews_corrcoef,
)
import numpy as np
import pandas as pd
from scipy.special import softmax
import matplotlib.pyplot as plt
import seaborn as sns

from happy.train.utils import (
    plot_confusion_matrix,
    plot_tissue_pr_curves,
    get_tissue_confusion_matrix,
)
from happy.models.graphsage import SupervisedSAGE, SupervisedDiffPool
from happy.models.clustergcn import ClusterGCN, ClusterGCNConvNet, JumpingClusterGCN
from happy.models.gat import GAT, GATv2
from projects.placenta.graphs.graphs.create_graph import get_nodes_within_tiles


def setup_node_splits(
    data,
    tissue_class,
    mask_unlabelled,
    include_validation=True,
    val_patch_files=None,
    test_patch_files=None,
):
    all_xs = data["pos"][:, 0]
    all_ys = data["pos"][:, 1]

    # Mark everything as training data first
    train_mask = torch.ones(data.num_nodes, dtype=torch.bool)
    data.train_mask = train_mask

    # Mask unlabelled data to ignore during training
    if mask_unlabelled and tissue_class is not None:
        unlabelled_inds = (tissue_class == 0).nonzero()[0]
        unlabelled_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        unlabelled_mask[unlabelled_inds] = True
        data.unlabelled_mask = unlabelled_mask
        train_mask[unlabelled_inds] = False
        data.train_mask = train_mask
        print(f"{len(unlabelled_inds)} nodes marked as unlabelled")

    # Split the graph by masks into training, validation and test nodes
    if include_validation:
        if val_patch_files is None:
            if test_patch_files is None:
                print("No validation patch provided, splitting nodes randomly")
                data = RandomNodeSplit(num_val=0.15, num_test=0.15)(data)
            else:
                print(
                    "No validation patch provided, splitting nodes randomly into "
                    "train and val and using test patch"
                )
                data = RandomNodeSplit(num_val=0.15, num_test=0)(data)
                test_node_inds = []
                for file in test_patch_files:
                    patches_df = pd.read_csv(file)
                    for row in patches_df.itertuples(index=False):
                        test_node_inds.extend(
                            get_nodes_within_tiles(
                                (row.x, row.y), row.width, row.height, all_xs, all_ys
                            )
                        )
                test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
                test_mask[test_node_inds] = True
                data.val_mask[test_node_inds] = False
                data.train_mask[test_node_inds] = False
                data.test_mask = test_mask
            if mask_unlabelled and tissue_class is not None:
                data.val_mask[unlabelled_inds] = False
                data.train_mask[unlabelled_inds] = False
                data.test_mask[unlabelled_inds] = False
        else:
            print("Splitting graph by validation patch")
            val_node_inds = []
            for file in val_patch_files:
                patches_df = pd.read_csv(file)
                for row in patches_df.itertuples(index=False):
                    if (
                        row.x == 0
                        and row.y == 0
                        and row.width == -1
                        and row.height == -1
                    ):
                        data.val_mask = torch.ones(data.num_nodes, dtype=torch.bool)
                        data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
                        data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
                        print("All nodes marked as validation")
                        return data
                    val_node_inds.extend(
                        get_nodes_within_tiles(
                            (row.x, row.y), row.width, row.height, all_xs, all_ys
                        )
                    )
            val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            val_mask[val_node_inds] = True
            train_mask[val_node_inds] = False
            if test_patch_files is not None:
                test_node_inds = []
                for file in test_patch_files:
                    patches_df = pd.read_csv(file)
                    for row in patches_df.itertuples(index=False):
                        test_node_inds.extend(
                            get_nodes_within_tiles(
                                (row.x, row.y), row.width, row.height, all_xs, all_ys
                            )
                        )
                test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
                test_mask[test_node_inds] = True
                train_mask[test_node_inds] = False
                data.test_mask = test_mask
            else:
                data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            data.val_mask = val_mask
            data.train_mask = train_mask
        print(
            f"Graph split into {data.train_mask.sum().item()} train nodes "
            f"and {data.val_mask.sum().item()} validation nodes "
            f"and {data.test_mask.sum().item()} test nodes"
        )
    else:
        data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    return data


def setup_dataloaders(
    model_type,
    data,
    num_layers,
    batch_size,
    num_neighbors,
):
    if (
        model_type == "sup_clustergcn"
        or model_type == "sup_gat"
        or model_type == "sup_jumping"
    ):
        cluster_data = ClusterData(
            data, num_parts=int(data.x.size()[0] / num_neighbors), recursive=False
        )
        train_loader = ClusterLoader(
            cluster_data, batch_size=batch_size, shuffle=True, num_workers=0
        )
        val_loader = NeighborSampler(
            data.edge_index, sizes=[-1], batch_size=1024, shuffle=False, num_workers=0
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
            hidden_channels=256,
            out_channels=num_classes,
            num_layers=layers,
        )
    elif model_type == "sup_gat":
        model = GAT(
            data.num_node_features,
            hidden_channels=256,
            out_channels=num_classes,
            heads=1,
            num_layers=layers,
        )
    elif model_type == "sup_clustergcn":
        model = ClusterGCN(
            data.num_node_features,
            hidden_channels=256,
            out_channels=num_classes,
            num_layers=layers,
        )
    elif model_type == "sup_jumping":
        model = JumpingClusterGCN(
            data.num_node_features,
            hidden_channels=256,
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
    elif (
        model_type == "sup_clustergcn"
        or model_type == "sup_gat"
        or model_type == "sup_jumping"
    ):
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

    if (
        model_type == "sup_clustergcn"
        or model_type == "sup_gat"
        or model_type == "sup_jumping"
    ):
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


def evaluate(tissue_class, predicted_labels, out, organ, run_path, remove_unlabelled):
    tissue_ids = [tissue.id for tissue in organ.tissues]
    if remove_unlabelled:
        tissue_ids = tissue_ids[1:]

    accuracy = accuracy_score(tissue_class, predicted_labels)
    balanced_accuracy = balanced_accuracy_score(tissue_class, predicted_labels)
    top_2_accuracy = top_k_accuracy_score(tissue_class, out, k=2, labels=tissue_ids)
    f1_macro = f1_score(tissue_class, predicted_labels, average="macro")
    cohen_kappa = cohen_kappa_score(tissue_class, predicted_labels)
    mcc = matthews_corrcoef(tissue_class, predicted_labels)
    roc_auc = roc_auc_score(
        tissue_class,
        softmax(out, axis=-1),
        average="macro",
        multi_class="ovo",
        labels=tissue_ids,
    )
    weighted_roc_auc = roc_auc_score(
        tissue_class,
        softmax(out, axis=-1),
        average="weighted",
        multi_class="ovo",
        labels=tissue_ids,
    )
    print("-----------------------")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Top 2 accuracy: {top_2_accuracy:.3f}")
    print(f"Balanced accuracy: {balanced_accuracy:.3f}")
    print(f"F1 macro score: {f1_macro:.3f}")
    print(f"Cohen's Kappa score: {cohen_kappa:.3f}")
    print(f"MCC score: {mcc:.3f}")
    print(f"ROC AUC macro: {roc_auc:.3f}")
    print(f"Weighted ROC AUC macro: {weighted_roc_auc:.3f}")
    print("-----------------------")

    print("Plotting confusion matrices")
    cm_df, cm_df_props = get_tissue_confusion_matrix(
        organ, predicted_labels, tissue_class, proportion_label=True
    )
    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1.1)
    plot_confusion_matrix(cm_df, "All Tissues", run_path, "d")
    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1.1)
    plot_confusion_matrix(cm_df_props, "All Tissues Proportion", run_path, ".2f")

    print("Plotting pr curves")
    tissue_mapping = {tissue.id: tissue.label for tissue in organ.tissues}
    tissue_colours = {tissue.id: tissue.colourblind_colour for tissue in organ.tissues}
    plot_tissue_pr_curves(
        tissue_mapping,
        tissue_colours,
        tissue_class,
        predicted_labels,
        out,
        run_path / "pr_curves.png",
    )


def collect_params(
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
    custom_weights,
):
    return pd.DataFrame(
        {
            "organ_name": organ_name,
            "exp_name": exp_name,
            "run_ids": [np.array(run_ids)],
            "x_min": x_min,
            "y_min": y_min,
            "width": width,
            "height": height,
            "k": k,
            "feature": feature,
            "graph_method": graph_method,
            "batch_size": batch_size,
            "num_neighbours": num_neighbours,
            "learning_rate": learning_rate,
            "weighted_loss": weighted_loss,
            "custom_weights": custom_weights,
            "epochs": epochs,
            "layers": layers,
        },
        index=[0],
    )


def save_state(run_path, logger, model, epoch):
    torch.save(model, run_path / f"{epoch}_graph_model.pt")
    logger.to_csv(run_path / "graph_train_stats.csv")


def _compute_tissue_weights(data_classes, organ, use_custom_weights):
    unique_classes = np.unique(data_classes)
    if not use_custom_weights:
        weighting = "balanced"
    else:
        custom_weights = [1, 0.85, 0.9, 10.5, 0.8, 1.3, 5.6, 3, 77]
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
