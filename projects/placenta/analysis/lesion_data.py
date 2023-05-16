import pandas as pd
import typer
import numpy as np

from happy.utils.utils import get_project_dir


LESIONS = [
    "intervillous_thrombos",
    "perivillous_fibrin",
    "infarction",
    "avascular_villi",
    "small_villi",
    "edemic",
    "inflammation",
    "null",
]


def main(
    project_name: str = typer.Option(...),
    csv_dir: str = typer.Option(...),
    features_data_csv: str = "",
):
    """Stats about the lesions file.

    Args:
        project_name: name of the project dir
        csv_dir: path from project dir to csvs with lesion data
        features_data_csv: name of csv with lesion features data
    """
    project_dir = get_project_dir(project_name)
    data_dir = project_dir / csv_dir
    plotting_features_df = pd.read_csv(data_dir / features_data_csv)

    print("-----------------------------------")
    print("Claudia certainty in slide:")
    ccert = plotting_features_df["claudia_certainty"]
    for certainty in ["obvious", "something", "healthy"]:
        certainty_count = (ccert == certainty).values.sum()
        print(f"{certainty}: {certainty_count}")

    print("-----------------------------------")
    print(f"Claudia diagnosis total lesions:")
    cdiag = plotting_features_df["claudia_diagnosis"]
    for lesion in LESIONS:
        if lesion == "null":
            null_values = pd.isna(cdiag)
            healthy = (null_values & (ccert == "healthy")).values.sum()
            print(f"unknown unhealthy: {null_values.values.sum() - healthy}")
        else:
            lesion_count = cdiag.str.contains(lesion).fillna(False).values.sum()
            print(f"{lesion}: {lesion_count}")

    print("-----------------------------------")
    print("Claudia certainty of lesion:")
    for lesion in LESIONS:
        if lesion == "null":
            pass
        else:
            lesion_series = cdiag.str.contains(lesion).fillna(False)
            obvious_lesion = ((ccert == "obvious") & lesion_series).values.sum()
            something_lesion = ((ccert == "something") & lesion_series).values.sum()
            print(
                f"{lesion}: obvious {obvious_lesion} and something {something_lesion}"
            )

    print("-----------------------------------")
    print("Claudia single or multi lesions on slide")
    for lesion in LESIONS:
        if lesion == "null":
            pass
        else:
            single_lesion = (cdiag == lesion).fillna(False).values.sum()
            multi_lesion = (
                cdiag.str.contains(lesion).fillna(False).values.sum() - single_lesion
            )
            print(f"{lesion}: single {single_lesion} and multi {multi_lesion}")

    print("-----------------------------------")
    print("Claudia lesion per trimester")
    trimesters = ["unknown", "1st_trimester", "2nd_trimester", "3rd_trimester", "term"]
    all_trimester_data = pd.cut(
        plotting_features_df["gestational_week"],
        [-2, 0, 13, 29, 40, np.inf],
        labels=trimesters,
    )
    for lesion in LESIONS:
        if lesion == "null":
            null_values = pd.isna(cdiag)
            healthy = null_values & (ccert == "healthy")
            print("unknown unhealthy")
            for trimester in trimesters:
                trimester_data = all_trimester_data == trimester
                lesion_trim_count = (
                    trimester_data & null_values & (ccert != "healthy")
                ).values.sum()
                print(f"{trimester}: {lesion_trim_count}")

            print("healthy")
            for trimester in trimesters:
                trimester_data = all_trimester_data == trimester
                lesion_trim_count = (trimester_data & healthy).values.sum()
                print(f"{trimester}: {lesion_trim_count}")
        else:
            print(lesion)
            lesion_data = cdiag.str.contains(lesion).fillna(False)
            for trimester in trimesters:
                trimester_data = all_trimester_data == trimester
                lesion_trim_count = (trimester_data & lesion_data).values.sum()
                print(f"{trimester}: {lesion_trim_count}")


if __name__ == "__main__":
    typer.run(main)
