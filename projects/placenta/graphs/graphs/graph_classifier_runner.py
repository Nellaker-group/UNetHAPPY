from dataclasses import dataclass, asdict
from typing import Optional
import json

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from happy.organs import Organ
from happy.graph.enums import GraphClassificationModelsArg
from happy.models.graph_classifier import TopKClassifer
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
    dropout: float
    learning_rate: float
    num_workers: int
    organ: Organ

    def save(self, seed, exp_name, run_path):
        to_save = {
            k: v for k, v in asdict(self).items() if k not in ("datasets", "organ")
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
        self.num_classes = len(self.params.organ.lesions)

    @staticmethod
    def new(params: Params, test: bool = False) -> "Runner":
        cls = {
            GraphClassificationModelsArg.top_k: TopKRunner,
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
        self._criterion = nn.NLLLoss()

    def _setup_dataloader(self):
        if not self.test:
            train_loader = DataLoader(
                self.params.datasets["train"],
                batch_size=self.params.batch_size,
                num_workers=self.params.num_workers,
                shuffle=True,
            )
            val_loader = DataLoader(
                self.params.datasets["val"],
                batch_size=self.params.batch_size,
                num_workers=self.params.num_workers,
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
            batch = batch.to(self.params.device)
            self.optimiser.zero_grad()

            out = self.model(batch)
            loss = self.criterion(out, batch.y)
            loss.backward()
            self.optimiser.step()
            pred = out.max(dim=1)[1]

            total_loss += loss.item() * batch.num_graphs
            total_correct += pred.eq(batch.y).sum().item()
        return total_loss / len(self.train_loader.dataset), total_correct / len(
            self.train_loader.dataset
        )

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        total_correct = 0
        for batch in self.val_loader:
            batch = batch.to(self.params.device)

            out = self.model(batch)
            loss = self.criterion(out, batch.y)
            loss.backward()
            pred = out.max(dim=1)[1]

            total_loss += loss.item() * batch.num_graphs
            total_correct += pred.eq(batch.y).sum().item()
        return total_loss / len(self.train_loader.dataset), total_correct / len(
            self.train_loader.dataset
        )

    def save_state(self, run_path, epoch):
        torch.save(self.model, run_path / f"{epoch}_c_graph_model.pt")


class TopKRunner(Runner):
    def setup_model(self):
        return TopKClassifer(
            self.params.datasets["train"][0].num_node_features,
            self.num_classes,
        )
