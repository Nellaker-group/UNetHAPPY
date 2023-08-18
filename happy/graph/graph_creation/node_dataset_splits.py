import pandas as pd
import torch
from torch_geometric.transforms import RandomNodeSplit

from happy.graph.graph_creation.create_graph import get_nodes_within_tiles


# Split nodes into unlabelled, training and validation sets. So far, validation
# and test sets are only defined for run_id 56 and 113. If there is training
# data in tissue_class for other listed runs, that data will also be used for training.
def setup_splits_by_runid(
    data, run_id, tissue_class, include_validation, val_patch_files, test_patch_files
):
    # this option will split the nodes in train, val, test based on patch files
    if run_id == 56 or run_id == 113:
        data = setup_node_splits(
            data,
            tissue_class,
            True,
            include_validation,
            val_patch_files,
            test_patch_files,
        )
    else:
        # this option will split the nodes randomly into train and val
        if include_validation and len(val_patch_files) == 0:
            data = setup_node_splits(data, tissue_class, True, include_validation=True)
        # this option will have no validation or test set and will use the whole graph
        else:
            data = setup_node_splits(data, tissue_class, True, include_validation=False)
    return data


def setup_node_splits(
    data,
    tissue_class,
    mask_unlabelled,
    include_validation=True,
    val_patch_files=[],
    test_patch_files=[],
    verbose=True,
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
        if verbose:
            print(f"{len(unlabelled_inds)} nodes marked as unlabelled")

    # Split the graph by masks into training, validation and test nodes
    if include_validation:
        if len(val_patch_files) == 0:
            if len(test_patch_files) == 0:
                if verbose:
                    print("No validation patch provided, splitting nodes randomly")
                data = RandomNodeSplit(num_val=0.15, num_test=0.15)(data)
            else:
                if verbose:
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
            if verbose:
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
                        if mask_unlabelled and tissue_class is not None:
                            data.val_mask[unlabelled_inds] = False
                            data.train_mask[unlabelled_inds] = False
                            data.test_mask[unlabelled_inds] = False
                        if verbose:
                            print(
                                f"All nodes marked as validation: "
                                f"{data.val_mask.sum().item()}"
                            )
                        return data
                    val_node_inds.extend(
                        get_nodes_within_tiles(
                            (row.x, row.y), row.width, row.height, all_xs, all_ys
                        )
                    )
            val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            val_mask[val_node_inds] = True
            train_mask[val_node_inds] = False
            if len(test_patch_files) > 0:
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
            if mask_unlabelled and tissue_class is not None:
                data.val_mask[unlabelled_inds] = False
                data.train_mask[unlabelled_inds] = False
                data.test_mask[unlabelled_inds] = False
        if verbose:
            print(
                f"Graph split into {data.train_mask.sum().item()} train nodes "
                f"and {data.val_mask.sum().item()} validation nodes "
                f"and {data.test_mask.sum().item()} test nodes"
            )
    else:
        data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    return data
