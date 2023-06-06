from happy.utils.utils import get_project_dir
from happy.organs import get_organ
from projects.placenta.graphs.graphs.graph_classification_utils import (
    setup_lesion_datasets,
)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    sns.set()
    project_dir = get_project_dir("placenta")
    organ = get_organ("placenta")
    lesion_map = {lesion.id: lesion.name for lesion in organ.lesions}

    # Get Dataset of lesion graphs (combination of single and multi lesions)
    datasets = setup_lesion_datasets(organ, project_dir, combine=True, test=False)
    test_dataset = setup_lesion_datasets(organ, project_dir, combine=True, test=True)
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
    df.plot(kind='bar', figsize=(10, 7))
    plt.title('Distribution of labels in each dataset')
    plt.xlabel('Labels')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # Check how many unique combinations are in each dataset
    train_unique_combinations = set(tuple(row) for row in train_dataset.lesions)
    val_unique_combinations = set(tuple(row) for row in val_dataset.lesions)
    test_unique_combinations = set(tuple(row) for row in test_dataset.lesions)
    print(f"Train uniques: {len(train_unique_combinations)}")
    print(f"Val uniques: {len(val_unique_combinations)}")
    print(f"Test uniques: {len(test_unique_combinations)}")


if __name__ == "__main__":
    main()
