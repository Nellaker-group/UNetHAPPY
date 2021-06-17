import collections

import pandas as pd

from happy.train.utils import plot_confusion_matrix


class Logger:
    def __init__(self, vis):
        self.vis = vis
        self.plot_to_vis = True if vis else False
        self.loss_hist = collections.deque(maxlen=500)

    # TODO: update the train_stats property here
    def log_accuracy(self, split_name, epoch_num, accuracy):
        print(f"{split_name} accuracy: {accuracy}")

        # self.train_stats[self.train_stats["epoch" == epoch_num]][
        #     f"{split_name}_accuracy"
        # ] = accuracy

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

    # TODO: update the train_stats property here
    def log_loss(self, split_name, epoch_num, loss):
        print(f"{split_name} loss: {loss}")
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

        for metric in metrics:
            for name in dataset_names:
                col = f"{name}_{metric}"
                columns.append(col)

        self.train_stats = pd.DataFrame(columns=columns)
