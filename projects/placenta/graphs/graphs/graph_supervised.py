import copy

import torch
from torch_geometric.loader import (
    ClusterData,
    ClusterLoader,
    NeighborSampler,
    NeighborLoader,
    DataLoader,
    GraphSAINTNodeSampler,
    GraphSAINTEdgeSampler,
    GraphSAINTRandomWalkSampler,
    ShaDowKHopSampler,
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    top_k_accuracy_score,
    f1_score,
    cohen_kappa_score,
    roc_auc_score,
    matthews_corrcoef,
    recall_score,
    precision_score,
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
from happy.models.graphsage import SupervisedSAGE
from happy.models.clustergcn import (
    ClusterGCN,
    JumpingClusterGCN,
    ClusterGCNEdges,
)
from models.gin import ClusterGIN
from happy.models.gat import GAT, GATv2
from happy.models.graphsaint import GraphSAINT
from happy.models.shadow import ShaDowGCN
from happy.models.sign import SIGN as SIGN_MLP
from happy.models.mlp import MLP


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
        or model_type == "sup_gatv2"
        or model_type == "sup_jumping"
        or model_type == "sup_clustergin"
        or model_type == "sup_clustergine"
        or model_type == "sup_clustergcn_w"
    ):
        cluster_data = ClusterData(
            data, num_parts=int(data.x.size()[0] / num_neighbors), recursive=False
        )
        train_loader = ClusterLoader(
            cluster_data, batch_size=batch_size, shuffle=True, num_workers=12
        )
        val_loader = NeighborSampler(
            data.edge_index, sizes=[-1], batch_size=1024, shuffle=False, num_workers=12
        )
    elif model_type.split("_")[1] == "graphsaint":
        sampler_type = model_type.split("_")[2]
        if sampler_type == "rw":
            train_loader = GraphSAINTRandomWalkSampler(
                data,
                batch_size=batch_size,
                walk_length=num_layers,
                num_steps=30,
                sample_coverage=num_neighbors,
                num_workers=12,
            )
        elif sampler_type == "node":
            train_loader = GraphSAINTNodeSampler(
                data,
                batch_size=batch_size,
                shuffle=True,
                num_steps=30,
                sample_coverage=num_neighbors,
                num_workers=12,
            )
        elif sampler_type == "edge":
            train_loader = GraphSAINTEdgeSampler(
                data,
                batch_size=batch_size,
                shuffle=True,
                num_steps=30,
                sample_coverage=num_neighbors,
                num_workers=12,
            )
        val_loader = NeighborSampler(
            data.edge_index, sizes=[-1], batch_size=1024, shuffle=False, num_workers=12
        )
    elif model_type == "sup_shadow":
        train_loader = ShaDowKHopSampler(
            data,
            depth=6,
            num_neighbors=num_neighbors,
            node_idx=data.train_mask,
            batch_size=batch_size,
            num_workers=12,
        )
        val_loader = ShaDowKHopSampler(
            data,
            depth=6,
            num_neighbors=num_neighbors,
            node_idx=None,
            batch_size=batch_size,
            num_workers=12,
            shuffle=False,
        )
    elif model_type == "sup_graphsage":
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
    elif model_type == "sup_sign" or model_type == "sup_mlp":
        train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
        val_idx = data.val_mask.nonzero(as_tuple=False).view(-1)

        train_loader = DataLoader(
            train_idx, batch_size=batch_size, shuffle=True, num_workers=12
        )
        val_loader = DataLoader(val_idx, batch_size=batch_size, shuffle=False)
    else:
        raise ValueError(f"Model type {model_type} not implemented")
    return train_loader, val_loader


def setup_model(
    model_type,
    data,
    device,
    layers,
    num_classes,
    hidden_units,
    dropout=None,
    pretrained=None,
):
    if pretrained:
        return torch.load(pretrained / "graph_model.pt", map_location=device)
    if model_type == "sup_graphsage":
        model = SupervisedSAGE(
            data.num_node_features,
            hidden_channels=hidden_units,
            out_channels=num_classes,
            num_layers=layers,
            dropout=dropout,
        )
    elif model_type == "sup_gat":
        model = GAT(
            data.num_node_features,
            hidden_channels=hidden_units,
            out_channels=num_classes,
            heads=4,
            num_layers=layers,
            dropout=dropout,
        )
    elif model_type == "sup_gatv2":
        model = GATv2(
            data.num_node_features,
            hidden_channels=hidden_units,
            out_channels=num_classes,
            heads=4,
            num_layers=layers,
            dropout=dropout,
        )
    elif model_type == "sup_clustergcn":
        model = ClusterGCN(
            data.num_node_features,
            hidden_channels=hidden_units,
            out_channels=num_classes,
            num_layers=layers,
            dropout=dropout,
            reduce_dims=64,
        )
    elif model_type == "sup_jumping":
        model = JumpingClusterGCN(
            data.num_node_features,
            hidden_channels=hidden_units,
            out_channels=num_classes,
            num_layers=layers,
            dropout=dropout,
        )
    elif model_type.split("_")[1] == "graphsaint":
        model = GraphSAINT(
            data.num_node_features,
            hidden_channels=hidden_units,
            out_channels=num_classes,
            num_layers=layers,
            dropout=dropout,
        )
    elif model_type == "sup_shadow":
        model = ShaDowGCN(
            data.num_node_features,
            hidden_channels=hidden_units,
            out_channels=num_classes,
            num_layers=layers,
            dropout=dropout,
        )
    elif model_type == "sup_sign":
        model = SIGN_MLP(
            data.num_node_features,
            hidden_channels=hidden_units,
            out_channels=num_classes,
            num_layers=layers,
            dropout=dropout,
        )
    elif model_type == "sup_mlp":
        model = MLP(
            data.num_node_features,
            hidden_channels=hidden_units,
            out_channels=num_classes,
            num_layers=layers,
            dropout=dropout,
        )
    elif model_type == "sup_clustergin":
        model = ClusterGIN(
            data.num_node_features,
            hidden_channels=hidden_units,
            out_channels=num_classes,
            num_layers=layers,
            dropout=dropout,
            reduce_dims=64,
        )
    elif model_type == "sup_clustergine":
        model = ClusterGIN(
            data.num_node_features,
            hidden_channels=hidden_units,
            out_channels=num_classes,
            num_layers=layers,
            dropout=dropout,
            reduce_dims=64,
            include_edge_attr=True,
        )
    elif model_type == "sup_clustergcn_w":
        model = ClusterGCNEdges(
            data.num_node_features,
            hidden_channels=hidden_units,
            out_channels=num_classes,
            num_layers=layers,
            dropout=dropout,
            reduce_dims=64,
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
    data=None,
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
    elif model_type == "sup_sign" or model_type == "sup_mlp":
        if weighted_loss:
            data_classes = data.y[data.train_mask].numpy()
            class_weights = _compute_tissue_weights(
                data_classes, organ, use_custom_weights
            )
            class_weights = torch.FloatTensor(class_weights)
            class_weights = class_weights.to(device)
            criterion = torch.nn.NLLLoss(weight=class_weights)
        else:
            criterion = torch.nn.NLLLoss()
    elif (
        model_type == "sup_clustergcn"
        or model_type == "sup_gat"
        or model_type == "sup_gatv2"
        or model_type == "sup_jumping"
        or model_type == "sup_clustergin"
        or model_type == "sup_clustergine"
        or model_type == "sup_clustergcn_w"
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
    elif model_type.split("_")[1] == "graphsaint" or model_type == "sup_shadow":
        if model_type.split("_")[1] == "graphsaint":
            reduction = "none"
        else:
            reduction = "mean"
        if weighted_loss:
            data_classes = train_dataloader.data.y[
                train_dataloader.data.train_mask
            ].numpy()
            class_weights = _compute_tissue_weights(
                data_classes, organ, use_custom_weights
            )
            class_weights = torch.FloatTensor(class_weights)
            class_weights = class_weights.to(device)
            criterion = torch.nn.NLLLoss(weight=class_weights, reduction=reduction)
        else:
            criterion = torch.nn.NLLLoss(reduction=reduction)
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
    data,
):
    model.train()

    total_loss = 0
    total_examples = 0
    total_correct = 0

    if (
        model_type == "sup_clustergcn"
        or model_type == "sup_gat"
        or model_type == "sup_gatv2"
        or model_type == "sup_jumping"
        or model_type == "sup_clustergin"
        or model_type == "sup_clustergine"
        or model_type == "sup_clustergcn_w"
    ):
        for batch in train_loader:
            batch = batch.to(device)
            optimiser.zero_grad()
            if (
                model_type == "sup_clustergine"
                or model_type == "sup_clustergcn_w"
                or model_type == "sup_clustergin"
            ):
                out = model(batch.x, batch.edge_index, batch.edge_attr)
            else:
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
    elif model_type.split("_")[1] == "graphsaint":
        model.set_aggr("add")
        for batch in train_loader:
            batch = batch.to(device)
            optimiser.zero_grad()
            edge_weight = batch.edge_norm * batch.edge_weight
            out = model(batch.x, batch.edge_index, edge_weight)
            loss = criterion(out, batch.y)
            loss = (loss * batch.node_norm)[batch.train_mask].sum()
            loss.backward()
            optimiser.step()

            nodes = batch.train_mask.sum().item()
            total_loss += loss.item() * nodes
            total_correct += int(
                out[batch.train_mask]
                .argmax(dim=-1)
                .eq(batch.y[batch.train_mask])
                .sum()
                .item()
            )
            total_examples += nodes
    elif model_type == "sup_shadow":
        for batch in train_loader:
            batch = batch.to(device)
            optimiser.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch, batch.root_n_id)
            loss = criterion(out, batch.y)
            loss.backward()
            optimiser.step()

            nodes = out.size()[0]
            total_loss += loss.item() * nodes
            total_correct += int(out.argmax(dim=-1).eq(batch.y).sum().item())
            total_examples += nodes
    elif model_type == "sup_graphsage":
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
    elif model_type == "sup_sign":
        for idx in train_loader:
            optimiser.zero_grad()
            train_x = [data.x[idx].to(device)]
            train_y = data.y[idx].to(device)
            train_x += [
                data[f"x{i}"][idx].to(device) for i in range(1, model.num_layers + 1)
            ]
            out = model(train_x)
            loss = criterion(out, train_y)
            loss.backward()
            optimiser.step()

            nodes = data.train_mask[idx].sum().item()
            total_loss += float(loss) * nodes
            total_correct += int((out.argmax(dim=-1).eq(train_y)).sum())
            total_examples += nodes
    elif model_type == "sup_mlp":
        for idx in train_loader:
            optimiser.zero_grad()
            train_x = data.x[idx].to(device)
            train_y = data.y[idx].to(device)
            out = model(train_x)
            loss = criterion(out, train_y)
            loss.backward()
            optimiser.step()

            nodes = data.train_mask[idx].sum().item()
            total_loss += float(loss) * nodes
            total_correct += int((out.argmax(dim=-1).eq(train_y)).sum())
            total_examples += nodes
    else:
        raise ValueError(f"No such model type {model_type}")

    return total_loss / total_examples, total_correct / total_examples


@torch.no_grad()
def validate(model, data, eval_loader, device):
    model.eval()
    if not isinstance(model, ShaDowGCN):
        if isinstance(model, GraphSAINT):
            model.set_aggr("mean")
        if isinstance(model, ClusterGIN) or isinstance(model, ClusterGCNEdges):
            out, _ = model.inference(data.x, data.edge_attr, eval_loader, device)
        else:
            out, _ = model.inference(data.x, eval_loader, device)
    else:
        out = []
        for batch in eval_loader:
            batch = batch.to(device)
            batch_out = model(batch.x, batch.edge_index, batch.batch, batch.root_n_id)
            out.append(batch_out)
        out = torch.cat(out, dim=0)
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
def validate_mlp(model, data, eval_loader, device):
    model.eval()
    out = []
    for idx in eval_loader:
        eval_x = data.x[idx].to(device)
        if isinstance(model, SIGN_MLP):
            eval_x = [eval_x]
            eval_x += [
                data[f"x{i}"][idx].to(device) for i in range(1, model.num_layers + 1)
            ]
        out_i, _ = model.inference(eval_x)
        out.append(out_i)
    out = torch.cat(out, dim=0)
    out = out.argmax(dim=-1)
    y = data.y.to(out.device)
    val_accuracy = int((out.eq(y[data.val_mask])).sum()) / int(data.val_mask.sum())
    return val_accuracy


@torch.no_grad()
def inference(model, data, eval_loader, device):
    print("Running inference")
    model.eval()
    if not isinstance(model, ShaDowGCN):
        if isinstance(model, GraphSAINT):
            model.set_aggr("mean")
        if isinstance(model, ClusterGIN) or isinstance(model, ClusterGCNEdges):
            out, graph_embeddings = model.inference(
                data.x, data.edge_attr, eval_loader, device
            )
        else:
            out, graph_embeddings = model.inference(data.x, eval_loader, device)
    else:
        out = []
        graph_embeddings = []
        for batch in eval_loader:
            batch = batch.to(device)
            batch_out, batch_embed = model.inference(
                batch.x, batch.edge_index, batch.batch, batch.root_n_id
            )
            out.append(batch_out)
            graph_embeddings.append(batch_embed)
        out = torch.cat(out, dim=0)
        graph_embeddings = torch.cat(graph_embeddings, dim=0)
    predicted_labels = out.argmax(dim=-1, keepdim=True).squeeze()
    predicted_labels = predicted_labels.cpu().numpy()
    out = out.cpu().detach().numpy()
    return out, graph_embeddings, predicted_labels


@torch.no_grad()
def inference_mlp(model, data, eval_loader, device):
    print("Running inference")
    model.eval()
    out = []
    graph_embeddings = []
    for idx in eval_loader:
        eval_x = data.x[idx].to(device)
        if isinstance(model, SIGN_MLP):
            eval_x = [eval_x]
            eval_x += [
                data[f"x{i}"][idx].to(device) for i in range(1, model.num_layers + 1)
            ]
        out_i, emb_i = model.inference(eval_x)
        out.append(out_i)
        graph_embeddings.append(emb_i)
    out = torch.cat(out, dim=0)
    graph_embeddings = torch.cat(graph_embeddings, dim=0)
    predicted_labels = out.argmax(dim=-1, keepdim=True).squeeze()
    predicted_labels = predicted_labels.cpu().numpy()
    out = out.cpu().detach().numpy()
    return out, graph_embeddings, predicted_labels


def evaluate(tissue_class, predicted_labels, out, organ, remove_unlabelled):
    tissue_ids = [tissue.id for tissue in organ.tissues]
    if remove_unlabelled:
        tissue_ids = tissue_ids[1:]

    accuracy = accuracy_score(tissue_class, predicted_labels)
    balanced_accuracy = balanced_accuracy_score(tissue_class, predicted_labels)
    top_2_accuracy = top_k_accuracy_score(tissue_class, out, k=2, labels=tissue_ids)
    top_3_accuracy = top_k_accuracy_score(tissue_class, out, k=3, labels=tissue_ids)
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
    print(f"Accuracy: {accuracy:.6f}")
    print(f"Top 2 accuracy: {top_2_accuracy:.6f}")
    print(f"Top 3 accuracy: {top_3_accuracy:.6f}")
    print(f"Balanced accuracy: {balanced_accuracy:.6f}")
    print(f"F1 macro score: {f1_macro:.6f}")
    print(f"Cohen's Kappa score: {cohen_kappa:.6f}")
    print(f"MCC score: {mcc:.6f}")
    print(f"ROC AUC macro: {roc_auc:.6f}")
    print(f"Weighted ROC AUC macro: {weighted_roc_auc:.6f}")
    print("-----------------------")


def evaluation_plots(tissue_class, predicted_labels, out, organ, run_path):
    # Order by counts and category
    sort_inds = [1, 2, 4, 0, 3, 5, 6, 7, 8]

    tissue_mapping = {tissue.id: tissue.name for tissue in organ.tissues}
    tissue_colours = {tissue.id: tissue.colour for tissue in organ.tissues}

    recalls = recall_score(tissue_class, predicted_labels, average=None)[sort_inds]
    precisions = precision_score(tissue_class, predicted_labels, average=None)[
        sort_inds
    ]
    print("Plotting recall and precision bar plots")
    plt.rcParams["figure.dpi"] = 600
    r_df = pd.DataFrame(recalls)
    plt.figure(figsize=(10, 3))
    sns.set(style="white", font_scale=1.2)
    colours = [tissue_colours[n] for n in np.unique(tissue_class)[sort_inds]]
    ax = sns.barplot(data=r_df.T, palette=colours)
    ax.set(ylabel="Recall", xticklabels=[])
    ax.tick_params(bottom=False)
    sns.despine(bottom=True)
    plt.savefig(run_path / "recalls.png")
    plt.close()
    plt.clf()

    p_df = pd.DataFrame(precisions)
    plt.rcParams["figure.dpi"] = 600
    plt.figure(figsize=(3, 10))
    sns.set(style="white", font_scale=1.2)
    ax = sns.barplot(data=p_df.T, palette=colours, orient="h")
    ax.set(xlabel="Precision", yticklabels=[])
    ax.tick_params(left=False)
    sns.despine(left=True)
    plt.savefig(run_path / "precisions.png")
    plt.close()
    plt.clf()

    print("Plotting tissue counts bar plot")
    _, tissue_counts = np.unique(tissue_class, return_counts=True)
    l_df = pd.DataFrame(tissue_counts[sort_inds])
    plt.rcParams["figure.dpi"] = 600
    plt.figure(figsize=(10, 3))
    sns.set(style="white", font_scale=1.2)
    ax = sns.barplot(data=l_df.T, palette=colours)
    ax.set(ylabel="Count", xticklabels=[])
    ax.tick_params(bottom=False)
    sns.despine(bottom=True)
    plt.savefig(run_path / "tissue_counts.png")
    plt.close()
    plt.clf()

    print("Plotting confusion matrices")
    cm_df, cm_df_props = get_tissue_confusion_matrix(
        organ, predicted_labels, tissue_class, proportion_label=False
    )
    sorted_labels = [tissue_mapping[n] for n in np.unique(tissue_class)[sort_inds]]
    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1)
    plot_confusion_matrix(cm_df, "All Tissues", run_path, "d", reorder=sorted_labels)
    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1)
    plot_confusion_matrix(
        cm_df_props, "All Tissues Proportion", run_path, ".2f", reorder=sorted_labels
    )

    print("Plotting pr curves")
    plot_tissue_pr_curves(
        tissue_mapping,
        tissue_colours,
        tissue_class,
        predicted_labels,
        out,
        run_path / "pr_curves.png",
    )


def collect_params(
    seed,
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
            "seed": seed,
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


def save_state(run_path, model, epoch):
    torch.save(model, run_path / f"{epoch}_graph_model.pt")


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
