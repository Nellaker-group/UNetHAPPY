import json

import typer
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score

from happy.utils.utils import get_project_dir
from projects.placenta.results.labelbox.path_mapping import PATHOLOGIST_MAPPING


def main(file_name: str = typer.Option(...)):
    project_dir = get_project_dir("placenta")
    path_to_file = project_dir / "results" / "labelbox" / file_name
    with open(path_to_file, "r") as f:
        data = json.load(f)

    cleaned_data = []
    for d in data:
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
    df = pd.DataFrame(cleaned_data)

    # Filter dataframe by whether labeller is a trained pathologist
    path_df = df[df["is_pathologist"] == 1]
    nonpath_df = df[df["is_pathologist"] == 0]

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
    all_predictions = get_images_and_labels_by_person(df, pathologist_only=True)

    kappa_0_1 = cohen_kappa_score(
        all_predictions[0]['label'], all_predictions[1]['label']
    )
    kappa_1_2 = cohen_kappa_score(
        all_predictions[1]['label'], all_predictions[2]['label']
    )
    kappa_0_2 = cohen_kappa_score(
        all_predictions[0]['label'], all_predictions[2]['label']
    )
    print(np.array([kappa_0_1, kappa_1_2, kappa_0_2]).mean())

    # TODO: total agreement between pathologists and non-experts
    # TODO: agreement for each tissue type between pathologists
    # TODO: agreement for each tissue type between pathologists and non-experts
    # TODO: unusual cases (most disagreement)
    # TODO: labels marked 'unclear'
    # TODO: Distribution of agreement per example (counts of same labels)
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
    for person in df['pathologist'].unique():
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

def get_num_unique_labels_per_image(df):
    return df.groupby('image_name')['label'].nunique()


if __name__ == "__main__":
    typer.run(main)
