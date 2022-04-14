import torch
from torch_geometric.utils.isolated import (
    contains_isolated_nodes,
    remove_isolated_nodes,
)
import numpy as np


def get_feature(feature, predictions, embeddings):
    if feature == "predictions":
        return predictions
    elif feature == "embeddings":
        return embeddings
    else:
        raise ValueError(f"No such feature {feature}")


def send_graph_to_device(data, device, tissue_class=None):
    x, edge_index = data.x.to(device), data.edge_index.to(device)
    if not tissue_class is None:
        tissue_class = torch.Tensor(tissue_class).type(torch.LongTensor).to(device)
    return x, edge_index, tissue_class


def save_model(model, save_path):
    torch.save(model, save_path)

# Similar to the function in microscopefile. May be merged at some point
def get_tile_coordinates(all_xs, all_ys, tile_width, tile_height):
    print(f"Tiling WSI into tiles of size {tile_width}x{tile_height}")
    # Find min x,y and max x,y from points and calculate num of rows and cols
    min_x, min_y = int(all_xs.min()), int(all_ys.min())
    max_x, max_y = int(all_xs.max()), int(all_ys.max())
    num_columns = int(np.ceil((max_x - min_x) / tile_width))
    num_rows = int(np.ceil((max_y - min_y) / tile_height))
    print(f"rows: {num_rows}, columns: {num_columns}")
    # Make list of minx and miny for each tile
    xy_list = []
    for col in range(num_columns):
        x = min_x + col * tile_width
        xy_list.extend([(x, y) for y in range(min_y, max_y + min_y, tile_height)])
    return xy_list


# Remove points from graph with edges which are too long (sort by edge length)
def remove_far_nodes(data, edge_max_length=25000):
    edge_lengths = data["edge_attr"].numpy().ravel()
    sorted_inds_over_length = (np.sort(edge_lengths) > edge_max_length).nonzero()[0]
    bad_edge_inds = np.argsort(data["edge_attr"].numpy().ravel())[
        sorted_inds_over_length
    ]
    print(
        f"Contains isolated nodes before edge removal? "
        f"{contains_isolated_nodes(data['edge_index'], data.num_nodes)}"
    )
    data["edge_index"] = _remove_element_by_indicies(data["edge_index"], bad_edge_inds)
    data["edge_attr"] = _remove_element_by_indicies(data["edge_attr"], bad_edge_inds)
    print(
        f"Contains isolated nodes after edge removal? "
        f"{contains_isolated_nodes(data['edge_index'], data.num_nodes)}"
    )
    print(
        f"Removed {len(bad_edge_inds)} edges "
        f"from graph with edge length > {edge_max_length}"
    )
    data["edge_index"], data["edge_attr"], mask = remove_isolated_nodes(
        data["edge_index"], data["edge_attr"], data.num_nodes
    )
    data["x"] = data["x"][mask]
    data["pos"] = data["pos"][mask]
    print(f"Removed {len(mask[mask == False])} isolated nodes")
    return data


def _remove_element_by_indicies(tensor, inds):
    if len(tensor) > 2:
        mask = torch.ones(tensor.size()[0], dtype=torch.bool)
        mask[inds] = False
        return tensor[mask]
    else:
        mask = torch.ones(tensor.size()[1], dtype=torch.bool)
        mask[inds] = False
        return tensor[:, mask]
