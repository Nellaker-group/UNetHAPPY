import csv

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from nucnet.utils.image_utils import load_image


class NucleiDataset(Dataset):
    """CSV dataset."""

    def __init__(
        self,
        annotations_dir,
        dataset_names,
        class_list_file,
        split="train",
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
        self.split = split
        self.transform = transform

        self.classes = self._load_classes()
        self.labels = self._load_labels()
        self.all_annotations = self._load_annotations()
        self.image_paths = list(self.all_annotations.keys())

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = load_image(self.image_paths[idx])
        annot = self.get_annotations_in_image(idx)
        sample = {"img": img, "annot": annot}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def get_annotations_in_image(self, image_index):
        # get ground truth annotations
        annotations_in_image = self.all_annotations[self.image_paths[image_index]]
        annotations = np.zeros((0, 5))

        # for images without annotations
        # this sets all of the empty annotations to an empty array of shape (0, 5)
        if len(annotations_in_image) == 0:
            return annotations

        # parse annotations
        for a in annotations_in_image:
            annotation = np.zeros((1, 5))

            annotation[0, 0] = a["x1"]
            annotation[0, 1] = a["y1"]
            annotation[0, 2] = a["x2"]
            annotation[0, 3] = a["y2"]
            annotation[0, 4] = self.classes[a["class"]]

            annotations = np.append(annotations, annotation, axis=0)

        return annotations

    # Reads each line of a csv and appends annotation information to each image. Loops through the desired datasets
    def _load_annotations(self):
        all_results = {}
        # This handles a single dataset name
        if isinstance(self.dataset_names, str):
            dataset_names = [self.dataset_names]
        else:
            dataset_names = self.dataset_names

        for dataset_name in dataset_names:
            file_path = f"{self.annotations_dir}/{dataset_name}/{self.split}_nuclei.csv"
            try:
                with open(file_path, "r", newline="") as file:
                    for line, row in enumerate(csv.reader(file, delimiter=",")):
                        line += 1

                        try:
                            img_file, x1, y1, x2, y2, class_name = row[:6]
                        except ValueError:
                            raise ValueError(
                                f"line {line}: format should be 'img_file,x1,y1,x2,y2,class_name' or "
                                f"'img_file,,,,,,'"
                            )
                        if img_file not in all_results:
                            all_results[img_file] = []

                        # Checks for non-empty tiles
                        if (x1, y1, x2, y2, class_name) == ("", "", "", "", ""):
                            continue
                        else:
                            x1 = self._string_to_int(x1, line)
                            y1 = self._string_to_int(y1, line)
                            x2 = self._string_to_int(x2, line)
                            y2 = self._string_to_int(y2, line)

                            # Check that the bounding box is valid.
                            if x2 <= x1:
                                raise ValueError(
                                    f"line {line}: x2 ({x2}) must be higher than x1 ({x1})"
                                )
                            if y2 <= y1:
                                raise ValueError(
                                    f"line {line}: y2 ({y2}) must be higher than y1 ({y1})"
                                )

                            # check if the current class name is correctly present
                            if class_name not in self.classes:
                                raise ValueError(
                                    f"line {line}: unknown class name: '{class_name}' (classes: {self.classes})"
                                )

                        all_results[img_file].append(
                            {
                                "x1": x1,
                                "x2": x2,
                                "y1": y1,
                                "y2": y2,
                                "class": class_name,
                            }
                        )
            except ValueError:
                raise ValueError(f"invalid csv annotations file: {file_path}")
        return all_results

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
                    class_id = self._string_to_int(class_id, line)
                    if class_name in result:
                        raise ValueError(
                            f"line {line}: duplicate class name: '{class_name}'"
                        )
                    # if it passes, add it to the results dictionary
                    result[class_name] = class_id
                return result
        except ValueError:
            raise ValueError(f"invalid csv class file: {self.class_list_file}")

    def _load_labels(self):
        labels = {}
        for key, value in self.classes.items():
            labels[value] = key
        return labels

    def num_classes(self):
        return max(self.classes.values()) + 1

    def image_aspect_ratio(self, image_index):
        try:
            image = Image.open(self.image_paths[image_index])
        except FileNotFoundError:
            image = Image.open("../" + self.image_paths[image_index])
        return float(image.width) / float(image.height)

    def _string_to_int(self, string, line):
        try:
            return int(string)
        except ValueError:
            raise ValueError(
                f"line {line}: malformed {string}, cannot convert to an integer"
            )
