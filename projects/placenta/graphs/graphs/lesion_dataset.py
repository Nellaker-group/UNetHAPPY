import torch
from torch_geometric.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer


class LesionDataset(Dataset):
    def __init__(
        self,
        organ,
        root,
        dataset_type,
        split,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.organ = organ
        self.split = split
        self.data_dir = root / "datasets" / "lesion" / dataset_type
        self.annotations_dir = root / "annotations" / "lesion" / dataset_type

        self.split_df = pd.read_csv(self.annotations_dir / f"{self.split}_lesion.csv")
        self.run_ids = (
            self.split_df.groupby("run_id")["lesion"].apply(list).index.to_numpy()
        )
        self.lesions = self._get_lesions()
        self.data_paths = [
            self.data_dir / f"run_{run_id}.pt" for run_id in self.run_ids
        ]
        super().__init__(root, transform, pre_transform, pre_filter)

    def len(self):
        return len(self.run_ids)

    def get(self, idx):
        data = torch.load(self.data_paths[idx])
        data.y = self.lesions[idx]
        return data

    def combine_with_other_dataset(self, second_dataset):
        self.split_df = pd.concat([self.split_df, second_dataset.split_df])
        self.run_ids = np.concatenate([self.run_ids, second_dataset.run_ids])
        self.lesions = np.concatenate([self.lesions, second_dataset.lesions])
        self.data_paths = np.concatenate([self.data_paths, second_dataset.data_paths])
        return self

    def _get_lesions(self):
        lesions = self.split_df.groupby("run_id")["lesion"].apply(list).to_numpy()
        mlb = MultiLabelBinarizer(
            classes=[lesion.label for lesion in self.organ.lesions]
        )
        multi_hot_encoded = mlb.fit_transform(lesions)
        return multi_hot_encoded
