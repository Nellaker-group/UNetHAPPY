import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import pandas as pd

from happy.hdf5.utils import get_datasets_in_patch, filter_by_confidence
from happy.models.graphsage import SAGE
from happy.hdf5.utils import get_embeddings_file
from projects.placenta.graphs.graphs.samplers.samplers import NeighborSampler
from projects.placenta.graphs.graphs.create_graph import (
    make_k_graph,
    make_delaunay_graph,
)


def get_raw_data(project_name, run_id, x_min, y_min, width, height, top_conf=False):
    embeddings_path = get_embeddings_file(project_name, run_id)
    print(f"Getting data from: {embeddings_path}")
    print(f"Using patch of size: x{x_min}, y{y_min}, w{width}, h{height}")
    # Get hdf5 datasets contained in specified box/patch of WSI
    predictions, embeddings, coords, confidence = get_datasets_in_patch(
        embeddings_path, x_min, y_min, width, height
    )
    if top_conf:
        predictions, embeddings, coords, confidence = filter_by_confidence(
            predictions, embeddings, coords, confidence, 0.9, 1.0
        )
    print(f"Data loaded with {len(predictions)} nodes")
    return predictions, embeddings, coords, confidence


def setup_graph(coords, k, feature, graph_method):
    data = Data(x=torch.Tensor(feature), pos=torch.Tensor(coords.astype("int32")))
    if graph_method == "k":
        graph = make_k_graph(data, k)
    elif graph_method == "delaunay":
        graph = make_delaunay_graph(data)
    else:
        raise ValueError(f"No such graph method: {graph_method}")
    if graph.x.ndim == 1:
        graph.x = graph.x.view(-1, 1)
    return graph


def setup_dataloader(data, batch_size, num_neighbors):
    return NeighborSampler(
        data.edge_index,
        sizes=[num_neighbors, num_neighbors],
        batch_size=batch_size,
        shuffle=True,
        num_nodes=data.num_nodes,
    )


def setup_model(data, device, layers):
    model = SAGE(data.num_node_features, hidden_channels=64, num_layers=layers)
    model = model.to(device)
    return model


def setup_parameters(data, model, learning_rate, device):
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    x, edge_index = data.x.to(device), data.edge_index.to(device)
    return optimiser, x, edge_index


def train(model, data, x, optimiser, train_loader, device):
    model.train()
    total_loss = 0
    for batch_size, n_id, adjs in train_loader:
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to(device) for adj in adjs]
        optimiser.zero_grad()

        out = model(x[n_id], adjs)
        out, pos_out, neg_out = out.split(out.size(0) // 3, dim=0)

        pos_loss = F.logsigmoid((out * pos_out).sum(-1)).mean()
        neg_loss = F.logsigmoid(-(out * neg_out).sum(-1)).mean()
        loss = -pos_loss - neg_loss
        loss.backward()
        optimiser.step()

        total_loss += float(loss) * out.size(0)
    return total_loss / data.num_nodes


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
    vis,
):
    torch.save(model.state_dict(), run_path / "graph_model.pt")

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
            "vis": vis,
        },
        index=[0],
    )
    params_df.to_csv(run_path / "params.csv", index=False)

    logger.to_csv(run_path / "graph_train_stats.csv")
