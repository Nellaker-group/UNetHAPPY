import torch
import torch.nn.functional as F
from torch_geometric.data import NeighborSampler
from torch_geometric.nn import DeepGraphInfomax
import pandas as pd

from happy.models.graphsage import SAGE
from happy.models.infomax import Encoder, corruption, summary_fn
from projects.placenta.graphs.graphs.samplers.samplers import (
    PosNegNeighborSampler,
)


def setup_dataloader(model_type, data, batch_size, num_neighbors):
    if model_type == "graphsage":
        return PosNegNeighborSampler(
            data.edge_index,
            sizes=[num_neighbors, num_neighbors],
            batch_size=batch_size,
            shuffle=True,
            num_nodes=data.num_nodes,
        )
    elif model_type == "infomax":
        return NeighborSampler(
            data.edge_index,
            node_idx=None,
            sizes=[10, 10, 25, 25],
            batch_size=256,
            shuffle=True,
            num_workers=12,
        )


def setup_model(model_type, data, device, layers=None, pretrained=None):
    if pretrained:
        return torch.load(pretrained / "graph_model.pt", map_location=device)
    if model_type == "graphsage":
        model = SAGE(data.num_node_features, hidden_channels=64, num_layers=layers)
    elif model_type == "infomax":
        model = DeepGraphInfomax(
            hidden_channels=512,
            encoder=Encoder(data.num_node_features, 512),
            summary=summary_fn,
            corruption=corruption,
        )
    else:
        return ValueError(f"No such model type implemented: {model_type}")
    model = model.to(device)
    return model


def setup_parameters(data, model, learning_rate, device):
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    x, edge_index = data.x.to(device), data.edge_index.to(device)
    return optimiser, x, edge_index


def train(model_type, model, data, x, optimiser, train_loader, device):
    model.train()
    total_loss = 0
    total_examples = 0
    for batch_size, n_id, adjs in train_loader:
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to(device) for adj in adjs]
        optimiser.zero_grad()

        if model_type == 'graphsage':
            out = model(x[n_id], adjs)
            out, pos_out, neg_out = out.split(out.size(0) // 3, dim=0)

            pos_loss = F.logsigmoid((out * pos_out).sum(-1)).mean()
            neg_loss = F.logsigmoid(-(out * neg_out).sum(-1)).mean()
            loss = -pos_loss - neg_loss
            total_loss += float(loss) * out.size(0)
            total_examples = data.num_nodes
        elif model_type == 'infomax':
            pos_z, neg_z, summary = model(x[n_id], adjs)
            loss = model.loss(pos_z, neg_z, summary)
            total_loss += float(loss) * pos_z.size(0)
            total_examples += pos_z.size(0)
        else:
            raise ValueError(f"No such model type implemented: {model_type}")

        loss.backward()
        optimiser.step()
    return total_loss / total_examples


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
    epochs,
    layers,
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
            "epochs": epochs,
            "layers": layers,
        },
        index=[0],
    )
    params_df.to_csv(run_path / "params.csv", index=False)

    logger.to_csv(run_path / "graph_train_stats.csv")


def _summary_lambda(z, *args, **kwargs):
    return torch.sigmoid(z.mean(dim=0))
