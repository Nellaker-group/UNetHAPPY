import collections

import pandas as pd

from happy.utils.vis_plotter import VisdomLinePlotter
from happy.train.utils import plot_confusion_matrix


class Logger:
    def __init__(self, vis):
        self.plot_to_vis = True if vis else False
        self.vis = VisdomLinePlotter() if vis else None
        self.loss_hist = collections.deque(maxlen=500)

    def log_accuracy(self, split_name, epoch_num, accuracy):
        print(f"{split_name} accuracy: {accuracy}")

        self._add_to_train_stats(epoch_num, split_name, accuracy)

        if self.plot_to_vis:
            self.vis.plot(
                "Accuracy",
                split_name,
                "Accuracy per Epoch (%)",
                "Epochs",
                "Accuracy (%)",
                epoch_num,
                accuracy,
            )

    def log_loss(self, split_name, epoch_num, loss):
        print(f"{split_name} loss: {loss}")

        self._add_to_train_stats(epoch_num, split_name, loss)

        if self.plot_to_vis:
            self.vis.plot(
                "loss",
                split_name,
                "Loss per Epoch",
                "Epochs",
                "Loss",
                epoch_num,
                loss,
            )

    def log_batch_loss(self, batch_count, loss):
        if self.plot_to_vis:
            self.vis.plot(
                "batch loss",
                "train",
                "Loss Per Batch",
                "Iteration",
                "Loss",
                batch_count,
                loss,
            )

    def log_confusion_matrix(self, cm, dataset_name, save_dir):
        print(cm)
        plot_confusion_matrix(cm, dataset_name, save_dir)

    def setup_train_stats(self, dataset_names, metrics):
        columns = ["epochs"]

        for name in dataset_names:
            for metric in metrics:
                col = f"{name}_{metric}"
                columns.append(col)

        self.train_stats = pd.DataFrame(columns=columns)

    def _add_to_train_stats(self, epoch_num, dataset_name, metric):
        column_name = f"{dataset_name}_{metric}"
        if not epoch_num in self.train_stats.index:
            row = pd.Series([metric], index=[column_name])
            self.train_stats = self.train_stats.append(row, ignore_index=True)
        else:
            self.train_stats.loc[epoch_num][column_name] = metric
