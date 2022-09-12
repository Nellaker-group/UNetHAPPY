import copy
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from torch_geometric.data import Data
from torch_geometric.loader import (
    DataLoader,
    ClusterData,
    ClusterLoader,
    NeighborSampler,
    NeighborLoader,
    GraphSAINTNodeSampler,
    GraphSAINTEdgeSampler,
    GraphSAINTRandomWalkSampler,
    ShaDowKHopSampler,
)
import torch
import torch.nn as nn

from happy.models.graphsage import SupervisedSAGE
from happy.models.clustergcn import ClusterGCN, JumpingClusterGCN
from happy.models.gat import GAT, GATv2
from happy.models.graphsaint import GraphSAINT
from happy.models.shadow import ShaDowGCN
from happy.organs.organs import Organ

from .enums import SupervisedModelsArg


@dataclass
class RunParams:
    data: Data
    device: str
    pretrained: Optional[str]
    model_type: SupervisedModelsArg
    batch_size: int
    num_neighbours: int
    epochs: int
    layers: int
    hidden_units: int
    learning_rate: float
    weighted_loss: bool
    use_custom_weights: bool
    include_validation: bool
    validation_step: int
    num_classes: int
    organ: Organ


@dataclass
class BatchResult:
    loss: torch.Tensor
    correct_predictions: int
    nodes: int


class ModelRunner:
    def __init__(self, params: RunParams):
        self.params: RunParams = params
        self._model: Optional[nn.Module] = None
        self._train_loader: Optional[DataLoader] = None
        self._val_loader: Optional[DataLoader] = None

    @staticmethod
    def new(params: RunParams) -> "ModelRunner":
        cls = {
            SupervisedModelsArg.sup_graphsage: GraphSAGERunner,
            SupervisedModelsArg.sup_clustergcn: ClusterGCNRunner,
            SupervisedModelsArg.sup_jumping: JumpingKnowledgeRunner,
            SupervisedModelsArg.sup_gat: GATRunner,
            SupervisedModelsArg.sup_graphsaint_rw: GraphSAINTRWRunner,
            SupervisedModelsArg.sup_graphsaint_edge: GraphSAINTEdgeRunner,
            SupervisedModelsArg.sup_graphsaint_node: GraphSAINTNodeRunner,
            SupervisedModelsArg.sup_shadow: ShaDowRunner,
        }[params.model_type]
        return cls(params)

    @property
    def model(self):
        if self._model is None:
            # If we we are using a pretrained model, load it from disk instead of
            # creating a new one.
            if self.params.pretrained is not None:
                self._model = torch.load(self.params.pretrained)
            else:
                self._model = self.setup_model()

        return self._model

    @property
    def train_loader(self):
        if self._train_loader is None:
            self._train_loader = self._setup_loaders()
        return self._train_loader

    @property
    def val_loader(self):
        if self._val_loader is None:
            self._val_loader = self._setup_loaders()
        return self._val_loader

    def _setup_loaders(self):
        self._train_loader, self._val_loader = self.setup_dataloader()

    @classmethod
    def setup_dataloader(cls):
        raise NotImplementedError(
            f"setup_dataloader not implemented for {cls.__name__}"
        )

    @classmethod
    def setup_model(cls):
        raise NotImplementedError(f"setup_model not implemented for {cls.__name__}")

    def setup_train_params(self):
        optimiser = torch.optim.Adam(
            self.model.parameters(), lr=self.params.learning_rate
        )
        criterion = self.setup_criterion()
        return optimiser, criterion

    @classmethod
    def setup_criterion(cls):
        """TODO This is only used by train, so we can just call it in there rather than at the top."""
        raise NotImplementedError(f"setup_criterion not implemented for {cls.__name__}")

    def train(self):
        optimiser, criterion = self.setup_train_params()
        self.model.train()

        total_loss = 0
        total_examples = 0
        total_correct = 0

        for batch in self.train_loader:
            batch = batch.to(self.params.device)
            optimiser.zero_grad()

            result = self.process_batch(optimiser, criterion, batch)

            total_loss += result.loss.item() * result.nodes
            total_correct += result.correct_predictions
            total_examples += result.nodes

    @classmethod
    def process_batch(cls, optimiser, criterion, batch) -> BatchResult:
        """Processes a batch and returns the number of correct predictions."""
        raise NotImplementedError(f"setup_criterion not implemented for {cls.__name__}")

    def validate(self):
        pass

    def inference(self):
        pass


class GraphSAGERunner(ModelRunner):
    def setup_dataloader(self):
        train_loader = NeighborLoader(
            self.params.data,
            input_nodes=self.params.data.train_mask,
            num_neighbors=[
                self.params.num_neighbours for _ in range(self.params.layers)
            ],
            batch_size=self.params.batch_size,
            shuffle=True,
            num_workers=12,
        )
        val_loader = NeighborLoader(
            copy.copy(self.params.data),
            num_neighbors=[-1],
            shuffle=False,
            batch_size=512,
        )
        val_loader.data.num_nodes = self.params.data.num_nodes
        val_loader.data.n_id = torch.arange(self.params.data.num_nodes)

        return train_loader, val_loader

    def setup_model(self):
        return SupervisedSAGE(
            self.params.data.num_node_features,
            hidden_channels=self.params.hidden_units,
            out_channels=self.params.num_classes,
            num_layers=self.params.layers,
        )

    def setup_criterion(self):
        if self.params.weighted_loss:
            data_classes = self.train_loader.data.y[
                self.train_loader.data.train_mask
            ].numpy()
            class_weights = _compute_tissue_weights(
                data_classes, self.params.organ, self.params.use_custom_weights
            )
            class_weights = torch.FloatTensor(class_weights)
            class_weights = class_weights.to(self.params.device)
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = torch.nn.CrossEntropyLoss()

        return criterion

    def process_batch(self, optimiser, criterion, batch) -> BatchResult:
        out = self.model(batch.x, batch.edge_index)
        train_out = out[batch.train_mask]
        train_y = batch.y[batch.train_mask]
        loss = criterion(train_out, train_y)
        loss.backward()
        optimiser.step()

        nodes = batch.train_mask.sum().item()
        return BatchResult(
            loss=loss,
            correct_predictions=int(train_out.argmax(dim=-1).eq(train_y).sum().item()),
            nodes=nodes,
        )


class ClusterGCNRunner(ModelRunner):
    def setup_dataloader(self):
        pass

    def setup_model(self):
        pass

    def setup_criterion(self):
        pass

    def train(self):
        pass


class JumpingKnowledgeRunner(ModelRunner):
    def setup_dataloader(self):
        pass

    def setup_model(self):
        pass

    def setup_criterion(self):
        pass

    def train(self):
        pass


class GATRunner(ModelRunner):
    def setup_dataloader(self):
        pass

    def setup_model(self):
        pass

    def setup_criterion(self):
        pass

    def train(self):
        pass


class GraphSAINTRWRunner(ModelRunner):
    def setup_dataloader(self):
        pass

    def setup_model(self):
        pass

    def setup_criterion(self):
        pass

    def train(self):
        pass


class GraphSAINTEdgeRunner(ModelRunner):
    def setup_dataloader(self):
        pass

    def setup_model(self):
        pass

    def setup_criterion(self):
        pass

    def train(self):
        pass


class GraphSAINTNodeRunner(ModelRunner):
    def setup_dataloader(self):
        pass

    def setup_model(self):
        pass

    def setup_criterion(self):
        pass

    def train(self):
        pass


class ShaDowRunner(ModelRunner):
    def setup_dataloader(self):
        pass

    def setup_model(self):
        pass

    def setup_criterion(self):
        pass

    def train(self):
        pass


def _compute_tissue_weights(data_classes, organ, use_custom_weights):
    unique_classes = np.unique(data_classes)
    if not use_custom_weights:
        weighting = "balanced"
    else:
        custom_weights = [1, 0.85, 0.9, 10.5, 0.8, 1.3, 5.6, 3, 77]
        weighting = dict(zip(list(unique_classes), custom_weights))
    class_weights = compute_class_weight(
        weighting, classes=unique_classes, y=data_classes
    )
    # Account for missing tissues in training data
    classes_in_training = set(unique_classes)
    all_classes = {tissue.id for tissue in organ.tissues}
    missing_classes = list(all_classes - classes_in_training)
    missing_classes.sort()
    for i in missing_classes:
        class_weights = np.insert(class_weights, i, 0.0)
    return class_weights
