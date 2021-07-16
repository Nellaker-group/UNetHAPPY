from abc import ABC, abstractmethod

import pandas as pd

from happy.utils.vis_plotter import VisdomLinePlotter
from happy.train.utils import plot_confusion_matrix


class _Appender(ABC):
    @abstractmethod
    def log_batch_loss(self, batch_count, loss):
        pass

    @abstractmethod
    def log_ap(self, split_name, epoch_num, ap):
        pass

    @abstractmethod
    def log_accuracy(self, split_name, epoch_num, accuracy):
        pass

    @abstractmethod
    def log_loss(self, split_name, epoch_num, loss):
        pass

    @abstractmethod
    def log_confusion_matrix(self, cm, dataset_name, save_dir):
        pass


class Console(_Appender):
    def log_batch_loss(self, batch_count, loss):
        pass

    def log_ap(self, split_name, epoch_num, ap):
        print(f"{split_name} AP: {ap}")

    def log_accuracy(self, split_name, epoch_num, accuracy):
        print(f"{split_name} accuracy: {accuracy}")

    def log_loss(self, split_name, epoch_num, loss):
        print(f"{split_name} loss: {loss}")

    def log_confusion_matrix(self, cm, dataset_name, save_dir):
        print(f"{dataset_name} confusion matrix:")
        print(cm)


class File(_Appender):
    def __init__(self, dataset_names, metrics):
        self.train_stats = self._setup_train_stats(dataset_names, metrics)

    def log_batch_loss(self, batch_count, loss):
        pass

    def log_ap(self, split_name, epoch_num, ap):
        self._add_to_train_stats(epoch_num, split_name, "AP", ap)

    def log_accuracy(self, split_name, epoch_num, accuracy):
        self._add_to_train_stats(epoch_num, split_name, "accuracy", accuracy)

    def log_loss(self, split_name, epoch_num, loss):
        self._add_to_train_stats(epoch_num, split_name, "loss", loss)
        
    def log_confusion_matrix(self, cm, dataset_name, save_dir):
        plot_confusion_matrix(cm, dataset_name, save_dir)

    def _setup_train_stats(self, dataset_names, metrics):
        columns = []
        for name in dataset_names:
            for metric in metrics:
                col = f"{name}_{metric}"
                columns.append(col)
        return pd.DataFrame(columns=columns)

    def _add_to_train_stats(self, epoch_num, dataset_name, metric_name, metric):
        column_name = f"{dataset_name}_{metric_name}"
        if not epoch_num in self.train_stats.index:
            row = pd.Series([metric], index=[column_name])
            self.train_stats = self.train_stats.append(row, ignore_index=True)
        else:
            self.train_stats.loc[epoch_num][column_name] = metric


class Visdom(_Appender):
    def __init__(self):
        self.plotter = VisdomLinePlotter()

    def log_batch_loss(self, batch_count, loss):
        self.plotter.plot(
            "batch loss",
            "train",
            "Loss Per Batch",
            "Iteration",
            "Loss",
            batch_count,
            loss,
        )

    def log_ap(self, split_name, epoch_num, ap):
        if split_name != "empty":
            self.plotter.plot(
                "AP",
                split_name,
                "AP per Epoch",
                "Epochs",
                "AP",
                epoch_num,
                ap,
            )

    def log_accuracy(self, split_name, epoch_num, accuracy):
        self.plotter.plot(
            "Accuracy",
            split_name,
            "Accuracy per Epoch (%)",
            "Epochs",
            "Accuracy (%)",
            epoch_num,
            accuracy,
        )

    def log_loss(self, split_name, epoch_num, loss):
        self.plotter.plot(
            "loss",
            split_name,
            "Loss per Epoch",
            "Epochs",
            "Loss",
            epoch_num,
            loss,
        )

    def log_confusion_matrix(self, cm, dataset_name, save_dir):
        pass
    