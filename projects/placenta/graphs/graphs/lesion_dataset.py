import os.path as osp

import torch
from torch_geometric.data import Dataset
import pandas as pd
import numpy as np


class LesionDataset(Dataset):
    def __init__(
        self, organ, root, split, transform=None, pre_transform=None, pre_filter=None
    ):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.organ = organ
        self.split = split

        self.split_df = pd.read_csv(osp.join(self.root, f"{self.split}_lesion.csv"))
        self.run_ids = self.split_df["run_id"].to_numpy()
        self.lesions = self.split_df["lesion"].to_numpy()
        self.classes = self._load_classes()

    def len(self):
        return len(self.run_ids)

    def get(self, idx):
        run_id = self.run_ids[idx]
        data = torch.load(osp.join(self.root, f"run_{run_id}.pt"))
        data.y = self.lesions[idx]
        return data

    def _load_classes(self):
        return {lesion.label: lesion.id for lesion in self.organ.lesions}

    def combine_with_other_dataset(self, second_dataset):
        self.split_df = pd.concat([self.split_df, second_dataset.split_df])
        self.run_ids = np.concatenate([self.run_ids, second_dataset.run_ids])
        self.lesions = np.concatenate([self.lesions, second_dataset.lesions])
        return self
