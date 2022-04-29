import torch
import torch.nn.functional as F
from torch_geometric.loader import ClusterData, ClusterLoader, NeighborLoader
import pandas as pd

from happy.models.graphsage import SupervisedSAGE, SupervisedDiffPool
from happy.models.clustergcn import ClusterGCN


def setup_dataloader(
    model_type,
    data,
    num_layers,
    batch_size,
    num_neighbors,
):
    if model_type == "sup_clustergcn":
        cluster_data = ClusterData(
            data, num_parts=int(data.x.size()[0] / num_neighbors), recursive=False
        )
        return ClusterLoader(
            cluster_data, batch_size=batch_size, shuffle=True, num_workers=12
        )
    else:
        return NeighborLoader(
            data,
            input_nodes=data.train_mask,
            num_neighbors=[num_neighbors for _ in range(num_layers)],
            batch_size=batch_size,
            shuffle=True,
            num_workers=12,
        )


def setup_model(model_type, data, device, layers, pretrained=None, num_classes=None):
    if pretrained:
        return torch.load(pretrained / "graph_model.pt", map_location=device)
    if model_type == "sup_graphsage":
        model = SupervisedSAGE(
            data.num_node_features,
            hidden_channels=64,
            out_channels=num_classes,
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
            hidden_channels=128,
            out_channels=num_classes,
            num_layers=layers,
        )
    else:
        return ValueError(f"No such model type implemented: {model_type}")
    model = model.to(device)
    return model


def train(
    model_type,
    model,
    optimiser,
    train_loader,
    device,
):
    model.train()

    total_loss = 0
    total_examples = 0
    total_correct = 0

    if model_type == "sup_clustergcn":
        for batch in train_loader:
            batch = batch.to(device)
            optimiser.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = F.nll_loss(out[batch.train_mask], batch.y[batch.train_mask])
            loss.backward()
            optimiser.step()

            nodes = batch.train_mask.sum().item()
            total_loss += loss.item() * nodes
            total_correct += int(out.argmax(dim=-1).eq(batch.y).sum().item())
            total_examples += nodes
    else:
        for batch in train_loader:
            batch = batch.to(device)
            optimiser.zero_grad()
            y = batch.y[: batch.batch_size]
            out = model(batch.x, batch.edge_index)[: batch.batch_size]
            loss = F.cross_entropy(out, y)
            loss.backward()
            optimiser.step()

            total_loss += float(loss) * batch.batch_size
            total_correct += int((out.argmax(dim=-1).eq(y)).sum())
            total_examples += batch.batch_size

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
