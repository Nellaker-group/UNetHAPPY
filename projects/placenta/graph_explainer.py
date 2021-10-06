import typer
import torch
from torch_geometric.nn import GNNExplainer
import matplotlib.pyplot as plt

from happy.utils.utils import get_device
from happy.utils.utils import get_project_dir
from graphs.graphs.create_graph import get_raw_data, setup_graph
from graphs.graphs.utils import get_feature
from graphs.graphs.enums import FeatureArg, MethodArg


def main(
    project_name: str = "placenta",
    exp_name: str = typer.Option(...),
    model_type: str = "graphsage",
    model_weights_dir: str = typer.Option(...),
    node_id: int = typer.Option(...),
    run_id: int = typer.Option(...),
    x_min: int = 0,
    y_min: int = 0,
    width: int = -1,
    height: int = -1,
    k: int = 6,
    feature: FeatureArg = FeatureArg.embeddings,
    graph_method: MethodArg = MethodArg.k,
    top_conf: bool = False,
):
    device = get_device()
    project_dir = get_project_dir(project_name)

    # Get data from hdf5 files
    predictions, embeddings, coords, confidence = get_raw_data(
        project_name, run_id, x_min, y_min, width, height, top_conf
    )

    feature_data = get_feature(feature.value, predictions, embeddings)
    data = setup_graph(coords, k, feature_data, graph_method.value)
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)

    # Setup trained model
    pretrained_path = (
        project_dir
        / "results"
        / "graph"
        / model_type
        / exp_name
        / model_weights_dir
        / "graph_model.pt"
    )
    model = torch.load(pretrained_path, map_location=device)

    # TODO: get this to work with graphsage. Right now the model's forward expects
    # TODO: adjs from the loader rather than the edge index
    explainer = GNNExplainer(model, epochs=50)
    node_feat_mask, edge_mask = explainer.explain_node(node_id, x, edge_index)
    ax, G = explainer.visualize_subgraph(node_id, edge_index, edge_mask)
    plt.savefig(project_dir)


if __name__ == "__main__":
    typer.run(main)
