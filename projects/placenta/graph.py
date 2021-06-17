from pathlib import Path
from enum import Enum

import typer
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.transforms import Distance
from torch_cluster import knn_graph
import umap
import umap.plot
import numpy as np

import happy.db.eval_runs_interface as db
from happy.utils.utils import print_gpu_stats
from happy.hdf5.utils import get_datasets_in_patch, filter_by_confidence
from happy.models.graphsage import SAGE
from happy.data.samplers.samplers import NeighborSampler
from happy.cells.cells import get_organ
from happy.utils.enum_args import OrganArg


class FeatureArg(str, Enum):
    predictions: "predictions"
    embeddings: "embeddings"


def main(
    run_id: int = typer.Option(...),
    x_min: int = 0,
    y_min: int = 0,
    width: int = -1,
    height: int = -1,
    k: int = 5,
    feature: FeatureArg = FeatureArg.embeddings,
    top_conf: bool = False,
    organ: OrganArg = OrganArg.placenta,
):
    organ = get_organ(organ.value)
    
    use_gpu = True  # a debug flag to allow non GPU testing of stuff. default true
    if use_gpu:
        print_gpu_stats()

    # Get data from hdf5 files
    predictions, embeddings, coords, confidence = get_raw_data(
        run_id, x_min, y_min, width, height, top_conf
    )

    if feature.value == "predictions":
        feature = predictions
    elif feature.value == "embeddings":
        feature = embeddings
    else:
        raise ValueError(f"No such feature {feature}")

    # Create the graph from the raw data
    data = setup_graph(coords, k, feature)

    # Setup the dataloader which minibatches the graph
    train_loader = NeighborSampler(
        data.edge_index,
        sizes=[10, 10],
        batch_size=256,
        shuffle=True,
        num_nodes=data.num_nodes,
    )

    # Setup the model
    model, optimiser, x, edge_index, device = setup_model(data)

    # Umap plot name
    conf_str = "_top_conf" if top_conf else ""
    plot_name = f"x{x_min}_y{y_min}_w{width}_h{height}{conf_str}"

    # Node embeddings before training
    plot_umap_embeddings(
        model, x, edge_index, predictions, plot_name, feature, False, organ.value
    )

    # Train!
    print("Training:")
    for epoch in range(1, 31):
        loss = train(model, data, x, optimiser, train_loader, device)
        print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")

    # Node embeddings after training
    plot_umap_embeddings(
        model, x, edge_index, predictions, plot_name, feature, True, organ.value
    )


def get_raw_data(run_id, x_min, y_min, width, height, top_conf=False):
    db.init()
    # Get path to data
    eval_run = db.get_eval_run_by_id(run_id)
    embeddings_root = Path(__file__).parent.parent / "results" / "embeddings"
    embeddings_path = eval_run.embeddings_path
    embeddings_dir = embeddings_root / embeddings_path
    print(f"Getting data from: {embeddings_path}")
    print(f"Using patch of size: x{x_min}, y{y_min}, w{width}, h{height}")
    # Get hdf5 datasets contained in specified box/patch of WSI
    predictions, embeddings, coords, confidence = get_datasets_in_patch(
        embeddings_dir, x_min, y_min, width, height
    )
    if top_conf:
        predictions, embeddings, coords, confidence = filter_by_confidence(
            predictions, embeddings, coords, confidence, 0.9, 1.0
        )
    print(f"Data loaded with {len(predictions)} nodes")
    return predictions, embeddings, coords, confidence


def setup_graph(coords, k, feature):
    print(f"Generating graph for k={k}")
    data = Data(x=torch.Tensor(feature), pos=torch.Tensor(coords.astype("int32")))
    data.edge_index = knn_graph(data.pos, k=k + 1, loop=True)
    get_edge_distance_weights = Distance(cat=False)
    data = get_edge_distance_weights(data)
    if data.x.ndim == 1:
        data.x = data.x.view(-1, 1)
    print(f"Graph made with {len(data.edge_index[0])} edges")
    return data


def setup_model(data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SAGE(data.num_node_features, hidden_channels=64, num_layers=2)
    model = model.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
    x, edge_index = data.x.to(device), data.edge_index.to(device)
    return model, optimiser, x, edge_index, device


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


def plot_umap_embeddings(
    model, x, edge_index, predictions, plot_name, feature, trained, organ
):
    embeddings = _get_graph_embeddings(model, x, edge_index)
    mapper = _fit_umap(embeddings)
    _plot_umap(mapper, predictions, plot_name, feature, trained, organ)


@torch.no_grad()
def _get_graph_embeddings(model, x, edge_index):
    print("Getting node embeddings for training data")
    model.eval()
    out = model.full_forward(x, edge_index).cpu()
    return out


def _fit_umap(embeddings):
    print("Fitting UMAP to embeddings")
    reducer = umap.UMAP(random_state=42, verbose=True, min_dist=0.1, n_neighbors=15)
    return reducer.fit(embeddings)


def _plot_umap(mapper, predictions, plot_name, feature, trained, organ):
    trained_str = "trained" if trained else "untrained"

    predictions_labelled = np.array([organ.by_id(pred).label for pred in predictions])
    colours_dict = {cell.label: cell.colour for cell in organ.cells}

    print("Plotting UMAP")
    plot = umap.plot.points(mapper, labels=predictions_labelled, color_key=colours_dict)
    save_dir = (
        Path(__file__).parent.parent
        / "visualisations"
        / "graphs"
        / "graphsage"
        / feature
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    plot_name = f"{trained_str}_{plot_name}.png"
    plot.figure.savefig(save_dir / plot_name)
    print(f"UMAP saved to {save_dir / plot_name}")


if __name__ == "__main__":
    typer.run(main)
