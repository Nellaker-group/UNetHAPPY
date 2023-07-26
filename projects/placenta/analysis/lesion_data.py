import pandas as pd
import typer
import numpy as np

from happy.utils.utils import get_project_dir
from happy.organs import get_organ


def main(
    project_name: str = typer.Option(...),
    organ_name: str = typer.Option(...),
    csv_dir: str = typer.Option(...),
    features_data_csv: str = "",
    diagnoser: str = "claudia",
):
    """Stats about the lesions file.

    Args:
        project_name: name of the project dir
        organ_name: name of the organ
        csv_dir: path from project dir to csvs with lesion data
        features_data_csv: name of csv with lesion features data
        diagnoser: name of the diagnoser
    """
    organ = get_organ(organ_name)
    lesions = [lesion.label for lesion in organ.lesions]
    project_dir = get_project_dir(project_name)
    data_dir = project_dir / csv_dir
    plotting_features_df = pd.read_csv(data_dir / features_data_csv)

    print("-----------------------------------")
    print(f"Claudia certainty in slide:")
    ccert = plotting_features_df[f"claudia_certainty"]
    for certainty in ["obvious", "something", "healthy"]:
        certainty_count = (ccert == certainty).values.sum()
        print(f"{certainty}: {certainty_count}")

    print("-----------------------------------")
    print(f"{diagnoser} diagnosis total lesions:")
    diag = plotting_features_df[f"{diagnoser}_diagnosis"]
    for lesion in lesions:
        if lesion == "null":
            null_values = pd.isna(diag)
            if diagnoser == "claudia":
                healthy = (null_values & (ccert == "healthy")).values.sum()
            else:
                healthy = null_values.values.sum()
            print(f"unknown unhealthy: {null_values.values.sum() - healthy}")
            print(f"47healthy: {healthy}")
        else:
            lesion_count = diag.str.contains(lesion).fillna(False).values.sum()
            print(f"{lesion}: {lesion_count}")

    print("-----------------------------------")
    print(f"Claudia certainty of lesion:")
    for lesion in lesions:
        if lesion == "null":
            pass
        else:
            cdiag = plotting_features_df[f"claudia_diagnosis"]
            lesion_series = cdiag.str.contains(lesion).fillna(False)
            obvious_lesion = ((ccert == "obvious") & lesion_series).values.sum()
            something_lesion = ((ccert == "something") & lesion_series).values.sum()
            print(
                f"{lesion}: obvious {obvious_lesion} and something {something_lesion}"
            )

    print("-----------------------------------")
    print(f"{diagnoser} single or multi lesions on slide")
    for lesion in lesions:
        if lesion == "null":
            pass
        else:
            single_lesion = (diag == lesion).fillna(False).values.sum()
            multi_lesion = (
                diag.str.contains(lesion).fillna(False).values.sum() - single_lesion
            )
            print(f"{lesion}: single {single_lesion} and multi {multi_lesion}")

    print("-----------------------------------")
    print(f"{diagnoser} lesion per trimester")
    trimesters = ["unknown", "1st_trimester", "2nd_trimester", "3rd_trimester", "term"]
    all_trimester_data = pd.cut(
        plotting_features_df["gestational_week"],
        [-2, 0, 13, 29, 40, np.inf],
        labels=trimesters,
    )
    for lesion in lesions:
        if lesion == "null":
            null_values = pd.isna(diag)
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
            lesion_data = diag.str.contains(lesion).fillna(False)
            for trimester in trimesters:
                trimester_data = all_trimester_data == trimester
                lesion_trim_count = (trimester_data & lesion_data).values.sum()
                print(f"{trimester}: {lesion_trim_count}")


if __name__ == "__main__":
    typer.run(main)
