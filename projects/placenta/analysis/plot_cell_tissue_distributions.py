from typing import Optional

import typer
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from happy.organs import get_organ
from happy.utils.utils import get_project_dir


def main(
    project_name: str = "placenta",
    organ_name: str = "placenta",
    data_dir: str = typer.Option(...),
    features_path: Optional[str] = None,
    single_cell_tissue: Optional[str] = None,
    plot_by_lesion: bool = False,
):
    """Plot the distribution of cell and tissue types across multiple WSIs."""
    organ = get_organ(organ_name)
    lesions = [lesion.label for lesion in organ.lesions]
    project_dir = get_project_dir(project_name)

    data_path = project_dir / data_dir
    cell_props = pd.read_csv(data_path / "cell_proportions.csv")
    tissue_props = pd.read_csv(data_path / "tissue_proportions.csv")
    cell_counts = pd.read_csv(data_path / "cell_counts.csv")
    tissue_counts = pd.read_csv(data_path / "tissue_counts.csv")

    if single_cell_tissue is None:
        # plot including everything
        _plot_all_slides(
            cell_props,
            tissue_props,
            cell_counts,
            tissue_counts,
            organ,
            data_path / "plots",
        )

        if plot_by_lesion:
            # plot filtered by lesion
            features_df = pd.read_csv(project_dir / features_path)
            for lesion in lesions:
                if lesion == "healthy":
                    lesion_run_ids = features_df[pd.isna(features_df["2nd_diagnosis"])][
                        "run_id"
                    ]
                else:
                    lesion_run_ids = features_df[
                        features_df["2nd_diagnosis"].str.contains(lesion).fillna(False)
                    ]["run_id"]

                lesion_cell_props = cell_props[cell_props["run_id"].isin(lesion_run_ids)]
                lesion_tissue_props = tissue_props[
                    tissue_props["run_id"].isin(lesion_run_ids)
                ]
                lesion_cell_counts = cell_counts[cell_counts["run_id"].isin(lesion_run_ids)]
                lesion_tissue_counts = tissue_counts[
                    tissue_counts["run_id"].isin(lesion_run_ids)
                ]

                lesion_save_dir = data_path / lesion / "plots"
                lesion_save_dir.mkdir(exist_ok=True, parents=True)
                _plot_all_slides(
                    lesion_cell_props,
                    lesion_tissue_props,
                    lesion_cell_counts,
                    lesion_tissue_counts,
                    organ,
                    lesion_save_dir,
                )
    else:
        features_df = pd.read_csv(project_dir / features_path)
        # filter the cell or tissue dfs by the single cell/tissue type
        if single_cell_tissue in cell_counts.columns:
            counts = cell_counts[["run_id", single_cell_tissue]]
        elif single_cell_tissue in tissue_counts.columns:
            counts = tissue_counts[["run_id", single_cell_tissue]]
        else:
            raise ValueError(
                f"{single_cell_tissue} is not a valid cell or tissue type."
            )
        features_df = features_df[["run_id", "2nd_diagnosis"]]
        lesion_counts = pd.merge(counts, features_df, on="run_id")
        lesion_counts.rename(columns={single_cell_tissue: "counts"}, inplace=True)
        lesion_counts.fillna("healthy", inplace=True)
        lesion_counts["2nd_diagnosis"] = (
            lesion_counts["2nd_diagnosis"].str.replace(", ", ",").str.split(",")
        )
        # Specific processing for KNT analysis
        if single_cell_tissue == "Syncytial Knot":
            lesion_counts["2nd_diagnosis"] = lesion_counts["2nd_diagnosis"].apply(
                lambda x: [
                    i
                    for i in x
                    if not (
                        i == "small_villi"
                        and (
                            "infarction" in x
                            or "intervillous_thrombos" in x
                            or "perivillous_fibrin" in x
                            or "inflammation" in x
                        )
                    )
                ]
            )
        lesion_counts = lesion_counts.explode("2nd_diagnosis")
        lesion_counts = lesion_counts[lesion_counts["2nd_diagnosis"].isin(lesions)]

        # Plot box with swarm of chosen cell or tissue type counts per lesion
        sns.set_style("white")
        plt.subplots(figsize=(8, 8), dpi=400)
        sns.boxplot(x="2nd_diagnosis", y="counts", data=lesion_counts)
        sns.swarmplot(
            x="2nd_diagnosis", y="counts", data=lesion_counts, color=".5", size=3
        )
        plt.title(f"Distribution of {single_cell_tissue} Density per Lesion")
        plt.xlabel("Lesion")
        plt.ylabel("Density")
        plt.xticks(rotation=90)
        plt.tight_layout()
        lesion_save_dir = data_path / "plots"
        lesion_save_dir.mkdir(exist_ok=True, parents=True)
        plt.savefig(lesion_save_dir / f"{single_cell_tissue}_counts.png")
        plt.close()
        plt.clf()


def plot_distribution(
    df,
    save_path,
    entity,
    colours,
    box=False,
    swarm=False,
    violin=False,
    cat=False,
    expectation=None,
    ylim=0.72,
    bottom=-0.02,
    ylabel=None,
):
    sns.set_style("white")
    plt.subplots(figsize=(8, 8), dpi=400)
    if box:
        ax = sns.boxplot(data=df, palette=colours, fliersize=1)
        if swarm:
            ax = sns.swarmplot(data=df, color=".5", size=3)
    elif swarm:
        ax = sns.swarmplot(data=df, palette=colours, size=1)
    elif violin:
        ax = sns.violinplot(data=df, palette=colours, bw=1.0, cut=0)
    elif cat:
        ax = sns.catplot(data=df, palette=colours, kind="violin")
    else:
        ax = sns.swarmplot(data=df, color=".5", size=1)
        _offset_swarm(ax, 0.3)

        melted_df = pd.melt(df.reset_index(drop=True).T.reset_index(), id_vars="index")
        ax = sns.lineplot(
            data=melted_df,
            x="index",
            y="value",
            hue="index",
            marker="o",
            err_style="bars",
            palette=colours,
            markersize=15,
            legend=False,
        )
        ax.lines[0].set_linestyle("")

    plt.ylim(bottom=bottom, top=ylim)
    ax.set(ylabel=ylabel)
    ax.set(xlabel=f"{entity} Labels")
    plt.xticks(rotation=90)
    plt.tight_layout()
    if expectation is not None:
        _add_expectation(ax, expectation)
    plt.savefig(save_path)
    plt.close()
    plt.clf()


def _offset_swarm(ax, offset):
    path_collections = [
        child
        for child in ax.get_children()
        if isinstance(child, matplotlib.collections.PathCollection)
    ]
    for path_collection in path_collections:
        x, y = np.array(path_collection.get_offsets()).T
        xnew = x + offset
        offsets = list(zip(xnew, y))
        path_collection.set_offsets(offsets)


def _add_expectation(ax, expectation):
    for i, (low, high) in enumerate(expectation):
        if low is not None:
            ax.hlines(
                y=low,
                xmin=i - 0.5,
                xmax=i + 0.5,
                color="lime",
                linestyles="dashed",
                linewidth=3,
            )
        if high is not None:
            ax.hlines(
                y=high,
                xmin=i - 0.5,
                xmax=i + 0.5,
                color="lime",
                linestyles="dashed",
                linewidth=3,
            )


def _plot_all_slides(
    cell_props,
    tissue_props,
    cell_counts,
    tissue_counts,
    organ,
    data_path,
):
    cell_colours = {cell.name: cell.colour for cell in organ.cells}
    cell_colours["Total"] = "#000000"
    tissue_colours = {tissue.name: tissue.colour for tissue in organ.tissues}

    cell_props = cell_props.drop("run_id", axis=1)
    plot_distribution(
        cell_props,
        data_path / "cell_proportions.png",
        "Cell",
        cell_colours,
        ylim=0.62,
        ylabel="Proportion of Cells Across WSIs",
        box=True,
        swarm=True,
    )
    tissue_props = tissue_props.drop("run_id", axis=1)
    plot_distribution(
        tissue_props,
        data_path / "tissue_proportions.png",
        "Tissue",
        tissue_colours,
        ylim=0.82,
        ylabel="Proportion of Tissues Across WSIs",
        box=True,
        swarm=True,
    )
    cell_counts = cell_counts.drop("run_id", axis=1)
    plot_distribution(
        cell_counts,
        data_path / "cell_counts.png",
        "Cell",
        cell_colours,
        ylim=6000.0,
        bottom=-200,
        ylabel="Number of Cells / mm^2",
        box=True,
        swarm=True,
    )
    tissue_counts = tissue_counts.drop("run_id", axis=1)
    tissue_counts = tissue_counts.drop("Total", axis=1)
    plot_distribution(
        tissue_counts,
        data_path / "tissue_counts.png",
        "Tissue",
        tissue_colours,
        ylim=3500.0,
        bottom=-200,
        ylabel="Number of Nuclei in Tissues / mm^2",
        box=True,
        swarm=True,
    )


TISSUE_EXPECTATION = [
    (0.0, 0.009),
    (0.30, 0.60),
    (0.17, 0.32),
    (None, None),
    (0.09, 0.25),
    (None, None),
    (None, None),
    (0.0, 0.10),
    (0.0, 0.0249),
]

if __name__ == "__main__":
    typer.run(main)