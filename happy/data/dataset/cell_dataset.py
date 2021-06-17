import csv
import os
from collections import defaultdict

from PIL import Image
from torch.utils.data import Dataset

from happy.utils.image_utils import load_image


class CellDataset(Dataset):
    def __init__(
        self,
        annotations_dir,
        dataset_names,
        class_list_file,
        split="train",
        oversampled_train=False,
        transform=None,
    ):
        """
        Args:
            annotations_dir (string): path to directory with dataset-specific annotation files
            dataset_names (list): list of directory names of datasets containing annotations files
            class_list_file (string): csv file with class list
            split (string): One of "train", "val", "test", "all" for choosing the correct annotations csv file
        """
        self.annotations_dir = annotations_dir
        self.dataset_names = dataset_names
        self.class_list_file = class_list_file
        self.oversampled_train = oversampled_train
        self.split = split
        self.transform = transform

        self.classes = self._load_classes()
        self.all_annotations = self._load_annotations()
        self.image_paths = [i[0] for i in self.all_annotations]

        self.class_sampling_weights = self._get_class_sampling_weights()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = load_image(self.image_paths[idx])
        annot = self.get_class_in_image(idx)
        sample = {"img": img, "annot": annot}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def get_class_in_image(self, image_index):
        # get ground truth annotations
        return self.classes[self.all_annotations[image_index][1]]

    def _load_annotations(self):
        all_results = []
        # This handles a single dataset name
        if isinstance(self.dataset_names, str):
            dataset_names = [self.dataset_names]
        else:
            dataset_names = self.dataset_names

        for dataset_name in dataset_names:
            # Get the file path and oversampled file if specified
            dir_path = f"{self.annotations_dir}/{dataset_name}"
            if (
                self.oversampled_train
                and self.split == "train"
                and os.path.isfile(f"{dir_path}/{self.split}_oversampled_cell.csv")
            ):
                file_path = f"{dir_path}/{self.split}_oversampled_cell.csv"
            else:
                file_path = (
                    f"{self.annotations_dir}/{dataset_name}/{self.split}_cell.csv"
                )

            try:
                with open(file_path, "r", newline="") as file:
                    for line, row in enumerate(csv.reader(file, delimiter=",")):
                        line += 1

                        try:
                            img_file, class_name = row[:2]
                        except ValueError:
                            raise ValueError(
                                f"line {line}: 'img_file,class_name' or 'img_file,,'"
                            )

                        # check if the current class name is correctly present
                        if class_name not in self.classes:
                            raise ValueError(
                                f"line {line}: unknown class name: "
                                f"'{class_name}' (classes: {self.classes})"
                            )

                        all_results.append([img_file, class_name])

            except ValueError:
                raise ValueError(f"invalid csv annotations file: {file_path}")
        return all_results

    # TODO: rather than pass in a csv file, pass in the organ
    def _load_classes(self):
        # parse the provided class file
        try:
            with open(self.class_list_file, "r", newline="") as file:
                result = {}
                for line, row in enumerate(csv.reader(file, delimiter=",")):
                    line += 1
                    # check that data in the csv file is well formed
                    try:
                        class_name, class_id = row
                    except ValueError:
                        raise ValueError(
                            f"line {line}: format should be 'class_name,class_id'"
                        )
                    class_id = int(class_id)
                    if class_name in result:
                        raise ValueError(
                            f"line {line}: duplicate class name: '{class_name}'"
                        )
                    # if it passes, add it to the results dictionary
                    result[class_name] = class_id
                return result
        except ValueError:
            raise ValueError(f"invalid csv class file: {self.class_list_file}")

    def _get_class_sampling_weights(self):
        class_list = [i[1] for i in self.all_annotations]
        class_counts = defaultdict(int)
        total_nr = 0
        for img_class in class_list:
            class_counts[img_class] += 1
            total_nr += 1
        assert total_nr == len(self)
        list_of_weights = [1 / class_counts[x] for x in class_list]
        return list_of_weights

    def num_classes(self):
        return max(self.classes.values()) + 1

    def image_aspect_ratio(self, image_index):
        image = Image.open(self.image_paths[image_index])
        return float(image.width) / float(image.height)
