import pandas as pd
import typer
import umap
import umap.plot
from bokeh.plotting import show, save, output_file
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import numpy as np

from happy.utils.utils import get_project_dir


def main(
    project_name: str = typer.Option(...),
    csv_dir: str = typer.Option(...),
    features_data_csv: str = "",
    umap_plot_name: str = "",
    value_of_interest: str = typer.Option(...),
    include_cells: bool = True,
    include_tissues: bool = False,
    include_proportions: bool = True,
    include_counts: bool = False,
    pca: bool = False,
    pca_variance: bool = False,
):
    """Plots and saves a UMAP or PCA from the cell and tissue distributions.

    Can be an interactive html file (open with browser or IDE) or simply a png.

    Args:
        project_name: name of the project dir to save to
        csv_dir: path from project dir to csvs with data
        features_data_csv: name of csv with additional features data
        umap_plot_name: name of html file to save to
        value_of_interest: name of the column for colouring the plot points
        include_cells: whether to include cell data
        include_tissues: whether to include tissue data
        include_proportions: whether to include proportion data
        include_counts: whether to include counts/density data
        pca: whether to use PCA instead of UMAP
        pca_variance: whether to plot the explained variance of the PCA
    """
    project_dir = get_project_dir(project_name)
    data_dir = project_dir / csv_dir

    # get run_ids
    run_ids = pd.read_csv(project_dir / "lesion_run_ids.csv", names=["run_id"])

    # Check for correct included data
    if not include_cells and not include_tissues:
        raise ValueError("Need to include at least cells or tissues")
    if not include_counts and not include_proportions:
        raise ValueError("Need to include at least count or proportion data")

    # Load in data
    df = run_ids
    cell_counts = pd.read_csv(data_dir / "cell_counts.csv")
    cell_proportions = pd.read_csv(data_dir / "cell_proportions.csv")
    tissue_counts = pd.read_csv(data_dir / "tissue_counts.csv")
    tissue_proportions = pd.read_csv(data_dir / "tissue_proportions.csv")

    # Combine data into one feature space
    if include_cells and include_counts:
        df = df.merge(cell_counts, on="run_id")
    if include_cells and include_proportions:
        df = df.merge(cell_proportions, on="run_id")
    if include_tissues and include_counts:
        df = df.merge(tissue_counts, on="run_id")
    if include_tissues and include_proportions:
        df = df.merge(tissue_proportions, on="run_id")

    # Prepare data for plotting colours
    ct_prop = cell_proportions.merge(tissue_proportions, on="run_id")
    cell_names = cell_proportions.columns[1:]
    tissue_names = tissue_proportions.columns[1:]
    if value_of_interest in cell_names or value_of_interest in tissue_names:
        data_of_interest = ct_prop[value_of_interest]
    elif value_of_interest == "Total":
        ct_counts = cell_counts.merge(tissue_counts, on="run_id")
        ct_counts["Total"] = cell_counts["Total"]
        ct_counts = ct_counts.drop(columns=["run_id"])
        data_of_interest = ct_counts[value_of_interest]
    else:
        plotting_features_df = pd.read_csv(data_dir / features_data_csv)
        if value_of_interest == "trimester":
            data_of_interest = pd.cut(
                plotting_features_df["gestational_week"],
                [-2, 0, 13, 29, 40, np.inf],
                labels=[
                    "unknown",
                    "1st_trimester",
                    "2nd_trimester",
                    "3rd_trimester",
                    "term",
                ],
            )
        elif "diagnosis" in value_of_interest:
            column_name = value_of_interest.split("_")[:2]
            column_name = "_".join(column_name)
            lesion = value_of_interest.split("_")[2:]
            lesion = "_".join(lesion)
            column_of_interest = plotting_features_df[column_name]
            data_of_interest = column_of_interest.str.contains(lesion).map(
                {True: lesion, False: ""}
            )
            data_of_interest = data_of_interest.fillna("")
        elif value_of_interest == "slide_name":
            data_of_interest = plotting_features_df["slide_name"]
            data_of_interest = data_of_interest.apply(lambda x: len(x) > 10).map(
                {True: "Estonia", False: "Israel"}
            )
        else:
            raise ValueError(f"Unknown value of interest: {value_of_interest}")

    df_features = df.drop(columns=["run_id"])
    ct_prop = ct_prop.drop(columns=["run_id"])

    if not pca:
        # Generate UMAP
        reducer = umap.UMAP(random_state=42, verbose=True)
        mapper = reducer.fit(df_features)

        plot = umap.plot.interactive(
            mapper,
            values=data_of_interest,
            interactive_text_search=True,
            hover_data=df[["run_id", "Terminal Villi", "Fibrin", "Avascular Villi"]],
            point_size=5,
        )

        output_file(data_dir / umap_plot_name, title=f"UMAP for {value_of_interest}")
        show(plot)
        print(f"saving interactive to {data_dir / umap_plot_name}")
        save(plot, data_dir / umap_plot_name)
    else:
        sns.set()
        # Generate PCA
        pca = PCA(n_components=5)
        pca_features = pca.fit_transform(df_features)
        pca_df = pd.DataFrame(data=pca_features)

        if pca_variance:
            # Plot explained variance
            plt.bar(range(1, len(pca.explained_variance_) + 1), pca.explained_variance_)
            plt.xlabel("PCA Feature")
            plt.ylabel("Explained variance")
            plt.title("Explained Variance")
            plt.show()

        # Setup directory and plot name
        data_dir = data_dir / "pca"
        if include_cells and not include_tissues:
            feature_type = "Cell"
            data_dir = data_dir / "cell_features"
        elif include_tissues and not include_cells:
            feature_type = "Tissue"
            data_dir = data_dir / "tissue_features"
        else:
            feature_type = "Cell_Tissue"
            data_dir = data_dir / "cell_tissue_features"

        # Prep total data for plotting
        ct_counts = cell_counts.merge(tissue_counts, on="run_id")
        ct_counts["Total"] = cell_counts["Total"]
        ct_counts = ct_counts.drop(columns=["run_id"])

        # Plot PCA coloured by each structure
        if features_data_csv is "":
            for structure in ct_prop.columns:
                data_of_interest = ct_prop[structure]
                plot_pca(pca_df, feature_type, data_of_interest, structure, data_dir)
            data_of_interest = ct_counts["Total"]
            plot_pca(pca_df, feature_type, data_of_interest, "Total", data_dir)
        else:
            plot_pca(
                pca_df, feature_type, data_of_interest, value_of_interest, data_dir
            )


# Plot PCA
def plot_pca(pca_df, feature_type, data_of_interest, value_of_interest, data_dir):
    sns.scatterplot(data=pca_df, x=0, y=1, hue=data_of_interest)
    # sns.scatterplot(data=pca_df, x=0, y=1, style=data_of_interest)
    plt.title(f"PCA for {value_of_interest} on {feature_type} features")
    value_name = value_of_interest.replace(" ", "_").replace("/", "_")
    plt.savefig(data_dir / f"{feature_type}_{value_name}_pca.png")
    print(
        f"PCA plot for {value_of_interest} using "
        f"{feature_type} features saved to {data_dir}"
    )
    plt.close()
    plt.clf()


if __name__ == "__main__":
    typer.run(main)
