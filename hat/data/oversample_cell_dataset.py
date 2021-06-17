import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter


def main():
    dataset_name = "towards"
    train_annot_path = f'../../Annotations/CellClass/{dataset_name}/train_cell.csv'
    train_df = pd.read_csv(train_annot_path, names=["file", "class"])
    x = train_df.iloc[:, 0]
    y = train_df.iloc[:, 1]

    print(f"Class distribution before sampling: {sorted(Counter(y).items())}")

    # sampling_strategy = {'CYT': 1300, 'FIB': 1300, 'HOF': 1000, 'SYN': 1539, 'VEN': 1300}
    sampling_strategy = {'CYT': 1000, 'FIB': 1500, 'HOF': 554, 'SYN': 1418, 'VEN': 1000}
    # ros = RandomOverSampler(random_state=0, sampling_strategy=sampling_strategy)
    ros = RandomUnderSampler(random_state=0, sampling_strategy=sampling_strategy)
    reshaped_x = x.to_numpy().reshape(-1, 1)
    x_resampled, y_resampled = ros.fit_resample(reshaped_x, y)
    x_resampled = x_resampled.reshape(-1)

    print(f"Class distribution after sampling: {sorted(Counter(y_resampled).items())}")
    assert x_resampled.shape == y_resampled.shape

    df = pd.DataFrame({"path": x_resampled, "class": y_resampled})

    save_dir = f"../../Annotations/CellClass/{dataset_name}/"
    df.to_csv(save_dir + 'train_oversampled_cell.csv', index=False, header=False)


if __name__ == '__main__':
    main()
