import collections

from happy.logger.appenders import Console, File, Visdom


class Logger:
    def __init__(self, vis, dataset_names, metrics):
        self.loss_hist = collections.deque(maxlen=500)
        self.appenders = self._get_appenders(vis, dataset_names, metrics)

    def log_ap(self, split_name, epoch_num, ap):
        for a in self.appenders:
            self.appenders[a].log_ap(split_name, epoch_num, ap)

    def log_accuracy(self, split_name, epoch_num, accuracy):
        for a in self.appenders:
            self.appenders[a].log_accuracy(split_name, epoch_num, round(accuracy, 4))

    def log_loss(self, split_name, epoch_num, loss):
        for a in self.appenders:
            self.appenders[a].log_loss(split_name, epoch_num, round(loss, 4))

    def log_batch_loss(self, batch_count, loss):
        for a in self.appenders:
            self.appenders[a].log_batch_loss(batch_count, round(loss, 4))

    def log_confusion_matrix(self, cm, dataset_name, save_dir):
        for a in self.appenders:
            self.appenders[a].log_confusion_matrix(cm, dataset_name, save_dir)

    def to_csv(self, save_path):
        file_appender = self.appenders['file']
        file_appender.train_stats.to_csv(save_path)

    def _get_appenders(self, vis, dataset_names, metrics):
        appenders = {'console': Console(), "file": File(dataset_names, metrics)}
        if vis:
            appenders["visdom"] = Visdom()
        return appenders
