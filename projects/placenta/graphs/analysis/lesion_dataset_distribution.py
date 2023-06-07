from typing import Optional, List

import typer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from happy.utils.utils import get_project_dir
from happy.organs import get_organ
from projects.placenta.graphs.graphs.graph_classification_utils import (
    setup_lesion_datasets,
)


def main(lesions_to_remove: Optional[List[str]] = None):
    print(f"Lesions to remove: {lesions_to_remove}")

    sns.set()
    project_dir = get_project_dir("placenta")
    organ = get_organ("placenta")
    lesion_map = {lesion.id: lesion.name for lesion in organ.lesions}

    # Get Dataset of lesion graphs (combination of single and multi lesions)
    datasets = setup_lesion_datasets(
        organ,
        project_dir,
        combine=True,
        test=False,
        lesions_to_remove=lesions_to_remove,
    )
    test_dataset = setup_lesion_datasets(
        organ, project_dir, combine=True, test=True, lesions_to_remove=lesions_to_remove
    )
    train_dataset = datasets["train"]
    val_dataset = datasets["val"]
    test_dataset = test_dataset["test"]

    # sum column-wise to get number of lesions per graph across the dataset
    train_sums = train_dataset.lesions.sum(axis=0)
    val_sums = val_dataset.lesions.sum(axis=0)
    test_sums = test_dataset.lesions.sum(axis=0)
    df = pd.DataFrame({"train": train_sums, "val": val_sums, "test": test_sums})
    df.index = [lesion_map[i] for i in df.index]

    # plot bar chart
    df.plot(kind="bar", figsize=(10, 7))
    plt.title("Distribution of labels in each dataset")
    plt.xlabel("Labels")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    # Check how many unique combinations are in each dataset
    train_unique_combinations = set(tuple(row) for row in train_dataset.lesions)
    val_unique_combinations = set(tuple(row) for row in val_dataset.lesions)
    test_unique_combinations = set(tuple(row) for row in test_dataset.lesions)
    print(f"Train uniques: {len(train_unique_combinations)}")
    print(f"Val uniques: {len(val_unique_combinations)}")
    print(f"Test uniques: {len(test_unique_combinations)}")

    # Check the frequency of label counts across datasets
    num_labels_per_instance_train = train_dataset.lesions.sum(axis=1)
    label_counts_train = np.bincount(num_labels_per_instance_train)
    for i, count in enumerate(label_counts_train):
        print(f"There are {count} instances with {i} labels in the training set.")
    num_labels_per_instance_val = val_dataset.lesions.sum(axis=1)
    label_counts_val = np.bincount(num_labels_per_instance_val)
    for i, count in enumerate(label_counts_val):
        print(f"There are {count} instances with {i} labels in the validation set.")
    num_labels_per_instance_test = test_dataset.lesions.sum(axis=1)
    label_counts_test = np.bincount(num_labels_per_instance_test)
    for i, count in enumerate(label_counts_test):
        print(f"There are {count} instances with {i} labels in the test set.")

    max_len = max(
        len(label_counts_train), len(label_counts_val), len(label_counts_test)
    )
    # Extend the arrays with zeros
    label_counts_train = np.pad(
        label_counts_train, (0, max_len - len(label_counts_train)), mode="constant"
    )
    label_counts_val = np.pad(
        label_counts_val, (0, max_len - len(label_counts_val)), mode="constant"
    )
    label_counts_test = np.pad(
        label_counts_test, (0, max_len - len(label_counts_test)), mode="constant"
    )

    # The label locations
    label_locs = np.arange(1, max_len)

    # Create the bars
    plt.figure(figsize=(12, 8))
    bar_width = 0.3
    rects1 = plt.bar(
        label_locs - bar_width, label_counts_train[1:], bar_width, label="Train"
    )
    rects2 = plt.bar(label_locs, label_counts_val[1:], bar_width, label="Val")
    rects3 = plt.bar(
        label_locs + bar_width, label_counts_test[1:], bar_width, label="Test"
    )

    # Plot
    plt.ylabel("Number of instances")
    plt.xlabel("Number of labels per instance")
    plt.title("Distribution of label counts in datasets")
    plt.xticks(label_locs)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    typer.run(main)
