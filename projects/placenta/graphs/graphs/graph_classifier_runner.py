from dataclasses import dataclass, asdict
from typing import Optional
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import numpy as np

from happy.graph.enums import GraphClassificationModelsArg
from happy.models.graph_classifier import TopKClassifer, SAGClassifer, ASAPClassifer
from projects.placenta.graphs.graphs.lesion_dataset import LesionDataset


@dataclass
class Params:
    datasets: dict[str, LesionDataset]
    device: str
    pretrained: Optional[str]
    model_type: GraphClassificationModelsArg
    batch_size: int
    epochs: int
    layers: int
    hidden_units: int
    pooling_ratio: float
    subsample_ratio: float
    learning_rate: float
    num_workers: int
    lesions_to_remove: Optional[list[str]]

    def save(self, seed, exp_name, run_path):
        to_save = {
            k: v for k, v in asdict(self).items() if k not in ("datasets")
        }
        to_save["seed"] = seed
        to_save["exp_name"] = exp_name
        with open(run_path / "train_params.json", "w") as f:
            json.dump(to_save, f, indent=2)


class Runner:
    def __init__(self, params: Params, test: bool = False):
        self.params: Params = params
        self.test = test
        self._model: Optional[nn.Module] = None
        self._train_loader: Optional[DataLoader] = None
        self._val_loader: Optional[DataLoader] = None
        self._optimiser: Optional[torch.optim.Optimizer] = None
        self._criterion: Optional[nn.Module] = None
        self.num_classes = next(iter(self.params.datasets.values())).get_num_classes()

    @staticmethod
    def new(params: Params, test: bool = False) -> "Runner":
        cls = {
            GraphClassificationModelsArg.top_k: TopKRunner,
            GraphClassificationModelsArg.sag: SAGRunner,
            GraphClassificationModelsArg.asap: ASAPRunner,
        }
        ModelClass = cls[params.model_type]
        return ModelClass(params, test)

    @property
    def model(self):
        if self._model is None:
            # If we are using a pretrained model, load it from disk instead of
            # creating a new one.
            if self.params.pretrained is not None:
                self._model = torch.load(self.params.pretrained)
            else:
                self._model = self.setup_model()
            self._model = self._model.to(self.params.device)
        return self._model

    @property
    def train_loader(self) -> DataLoader | None:
        if self._train_loader is None and not self.test:
            self._setup_loaders()
        return self._train_loader

    @property
    def val_loader(self) -> DataLoader:
        if self._val_loader is None:
            self._setup_loaders()
        return self._val_loader

    @property
    def optimiser(self):
        if self._optimiser is None:
            self._setup_optimiser()
        return self._optimiser

    @property
    def criterion(self):
        if self._criterion is None:
            self._criterion = self._setup_criterion()
        return self._criterion

    def _setup_loaders(self):
        self._train_loader, self._val_loader = self._setup_dataloader()

    def _setup_optimiser(self):
        self._optimiser = torch.optim.Adam(
            self.model.parameters(), lr=self.params.learning_rate
        )

    def _setup_criterion(self):
        return _FocalLoss()

    def _setup_dataloader(self):
        if not self.test:
            train_loader = DataLoader(
                self.params.datasets["train"],
                batch_size=self.params.batch_size,
                num_workers=self.params.num_workers,
                shuffle=True,
                persistent_workers=True,
            )
            val_loader = DataLoader(
                self.params.datasets["val"],
                batch_size=self.params.batch_size,
                num_workers=self.params.num_workers,
                persistent_workers=True,
            )
        else:
            train_loader = None
            val_loader = DataLoader(
                self.params.datasets["test"],
                batch_size=self.params.batch_size,
                num_workers=self.params.num_workers,
            )
        return train_loader, val_loader

    @classmethod
    def setup_model(cls):
        raise NotImplementedError(f"setup_model not implemented for {cls.__name__}")

    def train(self):
        self.model.train()
        total_loss = 0
        total_correct = 0
        for batch in self.train_loader:
            start = time.time()

            if self.params.subsample_ratio > 0.0:
                batch = self._subsample(batch)

            batch.y = torch.FloatTensor(np.array(batch.y))
            batch = batch.to(self.params.device)
            self.optimiser.zero_grad()

            out = self.model(batch)
            loss = self.criterion(out, batch.y)
            loss.backward()
            self.optimiser.step()

            pred_threshold = 0.5
            pred = ((torch.sigmoid(out)).gt(pred_threshold)).int()
            correct = (pred.eq(batch.y.int())).float()
            accuracy = correct.mean().item()

            print(f"batch loss: {loss.item():.4f} | batch acc {accuracy:.4f}")

            total_loss += loss.item() * batch.num_graphs
            total_correct += accuracy
            timer_end = time.time()
            print(f"time per batch: {timer_end - start:.4f}s ")
        return total_loss / len(self.train_loader.dataset), total_correct / len(
            self.train_loader
        )

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        total_correct = 0
        for batch in self.val_loader:
            start = time.time()
            if self.params.subsample_ratio > 0.0:
                batch = self._subsample(batch)

            batch.y = torch.FloatTensor(np.array(batch.y))
            batch = batch.to(self.params.device)

            out = self.model(batch)
            loss = self.criterion(out, batch.y)

            pred_threshold = 0.5
            pred = ((torch.sigmoid(out)).gt(pred_threshold)).int()
            correct = (pred.eq(batch.y.int())).float()
            accuracy = correct.mean().item()

            print(f"batch loss: {loss.item():.4f} | batch acc {accuracy:.4f}")

            total_loss += loss.item() * batch.num_graphs
            total_correct += accuracy
            timer_end = time.time()
            print(f"time per batch: {timer_end - start:.4f}s ")
        return total_loss / len(self.val_loader.dataset), total_correct / len(
            self.val_loader
        )

    def save_state(self, run_path, epoch):
        torch.save(self.model, run_path / f"{epoch}_c_graph_model.pt")

    def _subsample(self, batch):
        num_to_keep = int(batch.num_nodes * self.params.subsample_ratio)
        keep_indices = np.random.choice(
            np.arange(batch.num_nodes), num_to_keep, replace=False
        )
        return batch.subgraph(torch.LongTensor(keep_indices))


class TopKRunner(Runner):
    def setup_model(self):
        return TopKClassifer(
            next(iter(self.params.datasets.values())).num_node_features,
            self.params.hidden_units,
            self.num_classes,
            self.params.layers,
            self.params.pooling_ratio,
        )

class SAGRunner(Runner):
    def setup_model(self):
        return SAGClassifer(
            next(iter(self.params.datasets.values())).num_node_features,
            self.params.hidden_units,
            self.num_classes,
            self.params.layers,
            self.params.pooling_ratio,
        )

class ASAPRunner(Runner):
    def setup_model(self):
        return ASAPClassifer(
            next(iter(self.params.datasets.values())).num_node_features,
            self.params.hidden_units,
            self.num_classes,
            self.params.layers,
            self.params.pooling_ratio,
        )


class _FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(_FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        return F_loss.mean()
