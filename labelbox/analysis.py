import json
from itertools import combinations

import typer
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt

from happy.utils.utils import get_project_dir
from projects.placenta.results.labelbox.path_mapping import PATHOLOGIST_MAPPING


def main(
    file_name: str = typer.Option(...),
    original_file_name: str = typer.Option(...),
    original_subregion_file_name: str = typer.Option(...),
):
    # Process labelbox json file
    project_dir = get_project_dir("placenta")
    path_to_file = project_dir / "results" / "labelbox" / file_name
    with open(path_to_file, "r") as f:
        raw_data = json.load(f)
    df = _process_labelbox_data(raw_data)

    # process original annotation files
    path_to_original_file = project_dir / "results" / "labelbox" / original_file_name
    path_to_original_subregion_file = (
        project_dir / "results" / "labelbox" / original_subregion_file_name
    )
    with open(path_to_original_file, "r") as f:
        original_data = json.load(f)
    with open(path_to_original_subregion_file, "r") as f:
        original_subregion_data = json.load(f)
    original_df = _process_original_data(original_data, original_subregion_data)

    # Setup datastructures for analysis bellow
    path_df = df[df["is_pathologist"] == 1]
    nonpath_df = df[df["is_pathologist"] == 0]
    path_predictions = get_images_and_labels_by_person(df, pathologist_only=True)
    nonpath_predictions = get_images_and_labels_by_person(df, pathologist_only=False)

    # total average time per tissue type
    time_per_tissue = time_per_tissue_type(df)
    time_per_tissue_mean = time_per_tissue.mean()

    # average time per tissue type for pathologists
    path_time_per_tissue = time_per_tissue_type(path_df)
    path_time_per_tissue_mean = path_time_per_tissue.mean()

    # average time per tissue type for non experts
    nonpath_time_per_tissue = time_per_tissue_type(nonpath_df)
    nonpath_time_per_tissue_mean = nonpath_time_per_tissue.mean()

    # counts of tissue types across pathologists
    path_tissue_counts = path_df["label"].value_counts()

    # counts of tissue types across non experts
    nonpath_tissue_counts = nonpath_df["label"].value_counts()

    # total agreement between pathologists
    path_kappas = get_total_kappa(path_predictions)
    print(np.array(path_kappas).mean())

    # TODO: total agreement between pathologists and non-experts
    # TODO: agreement for each tissue type between pathologists
    # TODO: agreement for each tissue type between pathologists and non-experts
    # TODO: unusual cases (most disagreement)
    # TODO: labels marked 'unclear'

    # Patholgoist distribution of agreement per example (majority of same labels)
    path_majority_df = get_majority_labels(path_df)
    plt.figure(figsize=(15, 8))
    path_majority_df.plot.bar(x='image_name', y='num_unique')
    plt.savefig(project_dir / "results" / "labelbox" / 'bar_plot.png')


    # TODO: total agreement between pathologists and me
    # TODO: agreement for each tissue type between pathologists and me
    # TODO: agreement between pathologists but not me
    # TODO: agreement between pathologists and me
    # TODO: Hierarchy tissues and agreement based on that...
    # TODO: total agreement between my old qupath self and my current labelbox self

    # TODO: plotting of all of this


def time_per_tissue_type(df):
    return df.groupby("label")["duration"].mean()


def get_images_and_labels_by_person(df, pathologist_only):
    predictions = []
    for person in df["pathologist"].unique():
        if pathologist_only:
            if PATHOLOGIST_MAPPING[person] == 1:
                predictions.append(get_predictions_by_pathologist(df, person))
        else:
            predictions.append(get_predictions_by_pathologist(df, person))
    return predictions


def get_predictions_by_pathologist(df, pathologist):
    predicted_labels = df[df["pathologist"] == pathologist][["image_name", "label"]]
    predicted_labels.sort_values(by="image_name", ignore_index=True, inplace=True)
    return predicted_labels


def get_total_kappa(all_predictions):
    indices = list(range(len(all_predictions)))
    indices_combinations = list(combinations(indices, 2))
    kappas = []
    for comb in indices_combinations:
        kappas.append(
            cohen_kappa_score(
                all_predictions[comb[0]]["label"], all_predictions[comb[1]]["label"]
            )
        )
    return kappas


def get_majority_labels(df):
    unique_counts_df = df.groupby("image_name")["label"].nunique().reset_index(drop=False)
    unique_counts_df["majority_class"] = (
        df.groupby("image_name")["label"].agg(pd.Series.mode).values
    )
    unique_counts_df.loc[
        unique_counts_df["majority_class"].apply(lambda x: isinstance(x, np.ndarray))
    , "majority_class"] = "none"
    unique_counts_df.columns = ['image_name', 'num_unique', 'majority_class']
    unique_counts_df = unique_counts_df.sort_values(by=['majority_class'])
    return unique_counts_df


def _process_labelbox_data(raw_data):
    cleaned_data = []
    for d in raw_data:
        if not d["Skipped"]:
            cleaned_d = {
                "image_name": d["External ID"],
                "pathologist": d["Created By"],
                "label": d["Label"]["classifications"][0]["answer"]["value"],
                "duration": d["Seconds to Label"],
                "has_comment": d["Has Open Issues"],
                "is_pathologist": PATHOLOGIST_MAPPING[d["Created By"]],
            }
            cleaned_data.append(cleaned_d)
    return pd.DataFrame(cleaned_data)


def _process_original_data(original_raw_data, original_subset_raw_data):
    cleaned_original_data = []
    for d in original_raw_data:
        cleaned_d = {
            "image_name": d["image_name"],
            "label": d["tissue_type"],
        }
        cleaned_original_data.append(cleaned_d)
    original_df = pd.DataFrame(cleaned_original_data)

    cleaned_original_subregion_data = []
    for d in original_subset_raw_data:
        cleaned_d = {
            "image_name": d["image_name"],
            "label": d["tissue_type"],
        }
        cleaned_original_subregion_data.append(cleaned_d)
    original_subregion_df = pd.DataFrame(cleaned_original_subregion_data)
    return pd.concat([original_df, original_subregion_df])


if __name__ == "__main__":
    typer.run(main)
