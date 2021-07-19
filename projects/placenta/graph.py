from enum import Enum

import typer
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from happy.utils.utils import get_device
from happy.hdf5.utils import get_datasets_in_patch, filter_by_confidence
from happy.models.graphsage import SAGE
from happy.cells.cells import get_organ
from happy.hdf5.utils import get_embeddings_file
from happy.logger.logger import Logger
from graphs.graphs.samplers.samplers import NeighborSampler
from graphs.graphs.create_graph import make_k_graph, make_delaunay_graph
from graphs.graphs.embeddings import plot_umap_embeddings


class FeatureArg(str, Enum):
    predictions = "predictions"
    embeddings = "embeddings"


class MethodArg(str, Enum):
    k = "k"
    delaunay = "delaunay"


def main(
    project_name: str = 'placenta',
    organ_name: str = "placenta",
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

    organ = get_organ(organ_name)

    # Setup recording of stats per batch and epoch
    logger = Logger(list("train"), ["loss"], vis=vis, file=False)

    # Get data from hdf5 files
    predictions, embeddings, coords, confidence = get_raw_data(
        project_name, run_id, x_min, y_min, width, height, top_conf
    )

    if feature.value == "predictions":
        feature_data = predictions
    elif feature.value == "embeddings":
        feature_data = embeddings
    else:
        raise ValueError(f"No such feature {feature}")

    # Create the graph from the raw data
    data = setup_graph(coords, k, feature_data, graph_method.value)

    # Setup the dataloader which minibatches the graph
    train_loader = NeighborSampler(
        data.edge_index,
        sizes=[10, 10],
        batch_size=256,
        shuffle=True,
        num_nodes=data.num_nodes,
    )

    # Setup the model
    model, optimiser, x, edge_index = setup_model(data, device, layers)

    # Umap plot name
    conf_str = "_top_conf" if top_conf else ""
    plot_name = f"x{x_min}_y{y_min}_w{width}_h{height}{conf_str}"

    # Node embeddings before training
    plot_umap_embeddings(
        model, x, edge_index, predictions, plot_name, feature.value, False, organ
    )

    # Train!
    print("Training:")
    for epoch in range(1, epochs + 1):
        loss = train(model, data, x, optimiser, train_loader, device)
        logger.log_loss("train", epoch, loss)

    # Node embeddings after training
    plot_umap_embeddings(
        model, x, edge_index, predictions, plot_name, feature.value, True, organ
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


def setup_model(data, device, layers):
    model = SAGE(data.num_node_features, hidden_channels=64, num_layers=layers)
    model = model.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
    x, edge_index = data.x.to(device), data.edge_index.to(device)
    return model, optimiser, x, edge_index


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


if __name__ == "__main__":
    typer.run(main)
