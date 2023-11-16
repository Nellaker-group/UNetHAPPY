import os
from collections import defaultdict

import numpy as np
import typer
import torch
from torch_geometric.data import Data
from torch_geometric.nn.pool import fps
from torch_geometric.nn.unpool import knn_interpolate
from torch_geometric.nn import knn
from sklearn.metrics.pairwise import cosine_similarity

import happy.db.eval_runs_interface as db
from happy.utils.utils import get_device, get_project_dir
from happy.organs import get_organ
from happy.graph.utils.visualise_points import visualize_points
from happy.graph.graph_creation.get_and_process import get_hdf5_data
from happy.models.utils.custom_layers import KnnEdges, pool_one_hop


def main(
    project_name: str = "placenta",
    organ_name: str = "placenta",
    run_id: int = typer.Option(...),
    num_pooling_steps: int = typer.Option(...),
    subsample_ratio: float = 0.5,
    pooling_ratio: float = 0.5,
    pooling_method: str = "fps",
    plot_edges: bool = True,
    plot_downsampling: bool = True,
    plot_upsampling: bool = True,
    plot_node_relationships: bool = True,
    include_cells: bool = True,
    include_tissues: bool = True,
):
    assert include_cells or include_tissues

    db.init()
    device = get_device()
    project_dir = get_project_dir(project_name)
    organ = get_organ(organ_name)

    # Get hdf5 for the original cell and tissue labels
    hdf5_data = get_hdf5_data(project_name, run_id, 0, 0, -1, -1, tissue=True)
    if subsample_ratio > 0.0:
        hdf5_data, mask = hdf5_data.filter_randomly(subsample_ratio)
    cell_predictions = hdf5_data.cell_predictions
    tissue_predictions = hdf5_data.tissue_predictions

    # Get graph to infer over
    data_dir = project_dir / "datasets" / "lesion"
    data_file_name = f"run_{run_id}.pt"
    if os.path.exists(data_dir / "single" / data_file_name):
        data = torch.load(data_dir / "single" / data_file_name)
    elif os.path.exists(data_dir / "multi" / data_file_name):
        data = torch.load(data_dir / "multi" / data_file_name)
    else:
        raise ValueError(f"Could not find data file {data_file_name}")

    if subsample_ratio > 0.0:
        keep_indices = np.where(mask)[0]
        data = data.subgraph(torch.LongTensor(keep_indices))
    if not include_cells:
        data.x = data.x[:, :64]
    if not include_tissues:
        data.x = data.x[:, 64:]
    data = data.to(device)

    # Setup knn edge layer
    knn_edges = KnnEdges(start_k=6, k_increment=1, no_op=not plot_edges)

    # Apply pooling and edge reconstruction
    print(f"Applying pooling and knn edges if required")
    perms = []
    datas = [data]
    x = data.x
    pos = data.pos
    edge_index = data.edge_index
    for i in range(num_pooling_steps):
        if pooling_method == "fps":
            perm = fps(pos, ratio=pooling_ratio)
        elif pooling_method == "random":
            num_to_keep = int(pos.shape[0] * pooling_ratio)
            perm = torch.randperm(pos.shape[0])[:num_to_keep]
        elif pooling_method == "one_hop":
            perm = pool_one_hop(edge_index, pos.shape[0], 1000, 0.75, i)
        else:
            raise ValueError(f"Unknown pooling method {pooling_method}")
        perms.append(perm)
        x, pos, edge_index, _, _, _, _ = knn_edges(
            x, pos, edge_index, None, None, perm, None, i
        )
        datas.append(Data(x=x[perm], pos=pos, edge_index=edge_index))
        print(f"Finished pooling step {i+1}")

    # Get data back onto cpu for results saving
    data = data.to("cpu")
    edge_index = data.edge_index
    save_path = (
        project_dir
        / "visualisations"
        / "graphs"
        / f"{pooling_method}_pooling"
        / f"run_{run_id}"
    )
    save_path.mkdir(parents=True, exist_ok=True)

    # Visualise original graph
    plot_name = f"original_cells.png"
    if not os.path.exists(save_path / plot_name):
        print(f"Generating image for original graphs")
        colours_dict = {cell.id: cell.colour for cell in organ.cells}
        colours = [colours_dict[label] for label in cell_predictions]
        visualize_points(
            organ,
            save_path / plot_name,
            data.pos,
            colours=colours,
            width=int(data.pos[:, 0].max()) - int(data.pos[:, 0].min()),
            height=int(data.pos[:, 1].max()) - int(data.pos[:, 1].min()),
            edge_index=edge_index,
        )
    plot_name = f"original_tissues.png"
    if not os.path.exists(save_path / plot_name):
        colours_dict = {tissue.id: tissue.colour for tissue in organ.tissues}
        colours = [colours_dict[label] for label in tissue_predictions]
        visualize_points(
            organ,
            save_path / plot_name,
            data.pos,
            colours=colours,
            width=int(data.pos[:, 0].max()) - int(data.pos[:, 0].min()),
            height=int(data.pos[:, 1].max()) - int(data.pos[:, 1].min()),
            edge_index=edge_index,
        )

    # Visualise pooled graphs
    if plot_downsampling:
        for i, data in enumerate(datas[1:]):
            perm = perms[i].to("cpu")
            pos = data.pos.to("cpu")
            if plot_edges:
                edge_index = data.edge_index.to("cpu")
            else:
                pos = pos[perm]
                edge_index = None

            cell_predictions = cell_predictions[perm]
            tissue_predictions = tissue_predictions[perm]
            print(f"Generating image for pooling layer {i+1}")
            plot_name = f"pool_{i}_cells.png"
            colours_dict = {cell.id: cell.colour for cell in organ.cells}
            colours = np.array([colours_dict[label] for label in cell_predictions])
            visualize_points(
                organ,
                save_path / plot_name,
                pos,
                colours=colours,
                width=int(pos[:, 0].max()) - int(pos[:, 0].min()),
                height=int(pos[:, 1].max()) - int(pos[:, 1].min()),
                edge_index=edge_index,
            )
            plot_name = f"pool_{i}_tissues.png"
            colours_dict = {tissue.id: tissue.colour for tissue in organ.tissues}
            colours = np.array([colours_dict[label] for label in tissue_predictions])
            visualize_points(
                organ,
                save_path / plot_name,
                pos,
                colours=colours,
                width=int(pos[:, 0].max()) - int(pos[:, 0].min()),
                height=int(pos[:, 1].max()) - int(pos[:, 1].min()),
                edge_index=edge_index,
            )

    # Check that features can interpolate upwards
    datas.reverse()
    x = datas[0].x.to(device)
    for i, data in enumerate(datas):
        if i == len(datas) - 1:
            break
        pos_x = data.pos.to(device)
        pos_y = datas[i + 1].pos.to(device)
        x = knn_interpolate(x, pos_x, pos_y, k=6)
        print(f"Finished interpolating step {i+1}")

        if plot_upsampling:
            # compare reconstructed/interpolated x to the real x
            similarity = _get_similarity_of_interpolation(x, datas[i + 1].x)

            # visualise the similarity of the interpolated nodes to their real nodes
            visualize_points(
                organ,
                save_path / f"up_{i}_interp_similarity.png",
                pos_y.to("cpu"),
                colours=similarity,
                width=int(pos_y.to("cpu")[:, 0].max())
                - int(pos_y.to("cpu")[:, 0].min()),
                height=int(pos_y.to("cpu")[:, 1].max())
                - int(pos_y.to("cpu")[:, 1].min()),
            )

    if plot_upsampling:
        # Calculate similarity between interpolated nodes and their neighbours
        x = x.to("cpu").numpy()
        similarity_sums = defaultdict(float)
        neighbor_counts = defaultdict(int)
        edge_index = data.edge_index.to("cpu").numpy()
        for node1, node2 in edge_index.reshape(-1, 2):
            if node1 <= node2:
                similarity = cosine_similarity([x[node1]], [x[node2]])[0][0]
                similarity_sums[node1] += similarity
                similarity_sums[node2] += similarity
                neighbor_counts[node1] += 1
                neighbor_counts[node2] += 1
        average_similarities = {
            node: similarity_sums[node] / neighbor_counts[node]
            for node in neighbor_counts
        }
        similarities = np.array(
            [average_similarities[node] for node in sorted(average_similarities.keys())]
        )
        print(f"Finished calculating neighbour similarities")

        visualize_points(
            organ,
            save_path / f"avg_neighbour_similarity.png",
            data.pos.to("cpu"),
            colours=similarities,
            width=int(data.pos.to("cpu")[:, 0].max())
            - int(data.pos.to("cpu")[:, 0].min()),
            height=int(data.pos.to("cpu")[:, 1].max())
            - int(data.pos.to("cpu")[:, 1].min()),
        )

    # Check number of nodes after reduction at each depth
    print(f"Number of nodes after reduction at each depth:")
    for i, perm in enumerate(perms):
        print(f"Layer {i}: {perm.size(0)} nodes")

    if plot_node_relationships:
        # Visualise supernodes vs all remaining nodes at the lowest level
        datas.reverse()
        full_pos = datas[0].pos.to("cpu")
        small_pos = datas[-2].pos.to("cpu")
        node_mask = torch.zeros(small_pos.size(0), dtype=torch.bool, device=device)
        node_mask[perms[-1]] = True
        colours = ["red" if node_mask[i] else "grey" for i in range(small_pos.size(0))]
        visualize_points(
            organ,
            save_path / f"smallest_super_nodes_vs_children.png",
            small_pos,
            colours=colours,
            width=int(small_pos[:, 0].max()) - int(small_pos[:, 0].min()),
            height=int(small_pos[:, 1].max()) - int(small_pos[:, 1].min()),
            point_size=20,
        )

        # Visualise supernodes vs all remaining nodes at the highest level
        smallest_pos = datas[-1].pos.to("cpu")
        node_mask = torch.zeros(full_pos.size(0), dtype=torch.bool, device=device)
        # Expand dims for broadcasting
        large_expanded = full_pos.unsqueeze(1)  # shape becomes [n, 1, 2]
        small_expanded = smallest_pos.unsqueeze(0)  # shape becomes [1, m, 2]
        # Check for matches and reduce along the last dimension
        matches = (large_expanded.eq(small_expanded)).all(dim=-1)
        # Get the indices of matches from the larger tensor
        super_nodes = torch.where(matches)[0]
        node_mask[super_nodes] = True
        colours = ["red" if node_mask[i] else "grey" for i in range(full_pos.size(0))]
        visualize_points(
            organ,
            save_path / f"full_super_nodes_vs_children.png",
            full_pos,
            colours=colours,
            width=int(full_pos[:, 0].max()) - int(full_pos[:, 0].min()),
            height=int(full_pos[:, 1].max()) - int(full_pos[:, 1].min()),
        )

        # Visualise child to the smallest supernode relationship (approx by knn)
        nearest_super_node = knn(
            datas[-1].pos.to(device), datas[0].pos.to(device), k=1
        )[1]
        print("knn complete")
        visualize_points(
            organ,
            save_path / f"child_knn.png",
            full_pos,
            colours=nearest_super_node.to("cpu").numpy(),
            width=int(full_pos[:, 0].max()) - int(full_pos[:, 0].min()),
            height=int(full_pos[:, 1].max()) - int(full_pos[:, 1].min()),
        )

        # Visualise child to smallest supernode relationship (approx by knn traversal)
        datas.reverse()
        # Get the positions of each graph to be traversed
        graph_list = [data.pos.to(device) for data in datas]
        # Initial mapping is just a range, implying each node maps to itself initially
        current_mapping = torch.arange(graph_list[-1].size(0))

        # Traverse from the largest graph to the smallest graph
        for source_pos, target_pos in zip(graph_list[::-1][1:], graph_list[::-1][:-1]):
            nearest_nodes_in_target_for_source = knn(source_pos, target_pos, k=1)[1]
            current_mapping = nearest_nodes_in_target_for_source[
                current_mapping
            ].squeeze()

        print("knn complete")
        full_pos = datas[-1].pos.to("cpu")
        visualize_points(
            organ,
            save_path / f"child_knn_full_traversal.png",
            full_pos,
            colours=current_mapping.to("cpu").numpy(),
            width=int(full_pos[:, 0].max()) - int(full_pos[:, 0].min()),
            height=int(full_pos[:, 1].max()) - int(full_pos[:, 1].min()),
        )


def _get_similarity_of_interpolation(interp_x, real_x):
    real_x = real_x.to("cpu").numpy()
    interp_x = interp_x.to("cpu").numpy()
    similarity = np.empty(real_x.shape[0])
    for node_ind in range(real_x.shape[0]):
        similarity[node_ind] = cosine_similarity(
            [interp_x[node_ind]], [real_x[node_ind]]
        )
    return similarity


if __name__ == "__main__":
    typer.run(main)
