import typer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from happy.organs.organs import get_organ


def main():
    organ = get_organ("placenta")

    cell_colours = np.array([cell.colourblind_colour for cell in organ.cells])
    cell_colours = cell_colours.take([3, 0, 4, 6, 1, 7, 2, 10, 9, 5, 8])
    cell_prop = {
        "SYN": 38.3,
        "CYT": 7.2,
        "VEN": 16.3,
        "VMY": 12.3,
        "FIB": 19.6,
        "WBC": 1.0,
        "HOF": 0.6,
        "KNT": 1.5,
        "EVT": 2.5,
        "MAT": 0.3,
        "MES": 0.3,
    }
    cell_prop = pd.Series(cell_prop)
    plot_dist(cell_prop, cell_colours, "cell_prop")
    tissue_colours = np.array([tissue.colourblind_colour for tissue in organ.tissues])
    tissue_colours = np.delete(tissue_colours, [0, 2, 4, 10])
    tissue_colours = tissue_colours.take([5, 4, 2, 1, 0, 3, 8, 7, 6])
    all_labelled_tissue_prop = {
        "Chorion": 7.4,
        "SVilli": 18.9,
        "MIVilli": 26.5,
        "TVilli": 37.9,
        "Sprout": 3.0,
        "AVilli": 1.1,
        "Avascular": 0.3,
        "Fibrin": 2.8,
        "Maternal": 1.9,
    }
    all_labelled_tissue_prop = pd.Series(all_labelled_tissue_prop)
    plot_dist(all_labelled_tissue_prop, tissue_colours, "tissue_prop")
    train_prop = {
        "Chorion": 9.94,
        "SVilli": 28.18,
        "MIVilli": 17.27,
        "TVilli": 34.01,
        "Sprout": 3.32,
        "AVilli": 1.23,
        "Avascular": 0.19,
        "Fibrin": 3.03,
        "Maternal": 2.84,
    }
    train_prop = pd.Series(train_prop)
    plot_dist(train_prop, tissue_colours, "train_tissue_prop")
    val_prop = {
        "Chorion": 2.40,
        "SVilli": 28.44,
        "MIVilli": 19.05,
        "TVilli": 40.75,
        "Sprout": 4.40,
        "AVilli": 0.79,
        "Avascular": 0.26,
        "Fibrin": 3.42,
        "Maternal": 0.50,
    }
    val_prop = pd.Series(val_prop)
    plot_dist(val_prop, tissue_colours, "val_tissue_prop")
    test_prop = {
        "Chorion": 2.20,
        "SVilli": 27.12,
        "MIVilli": 20.84,
        "TVilli": 40.43,
        "Sprout": 3.60,
        "AVilli": 0.92,
        "Avascular": 0.22,
        "Fibrin": 2.80,
        "Maternal": 1.88,
    }
    test_prop = pd.Series(test_prop)
    plot_dist(test_prop, tissue_colours, "test_tissue_prop")

    tissues_df = pd.concat([train_prop, val_prop, test_prop], axis=1)
    tissues_df.rename(columns={0: "train", 1: "val", 2: "test"}, inplace=True)
    fig = px.bar(tissues_df, barmode="group")
    fig.for_each_trace(lambda trace: trace.update(text=trace.name))
    fig.update_layout(plot_bgcolor="white")
    fig.update_traces(
        marker_color=tissue_colours,
        showlegend=False,
        textposition="outside",
    )
    fig.update_xaxes(showline=True, linewidth=1, linecolor="black")
    fig.update_yaxes(showline=True, linewidth=1, linecolor="black")
    fig.show()


def plot_dist(tissue_prop, tissue_colours, plot_name):
    sns.set(style="white")
    tissue_prop.plot(
        kind="bar",
        legend=False,
        width=0.8,
        figsize=(8.5, 6),
        ylim=[0, 40],
        color=tissue_colours,
    )
    sns.despine()
    plt.gcf().tight_layout()
    plt.savefig(f"plots/{plot_name}.png")
    plt.close()
    plt.clf()


if __name__ == "__main__":
    typer.run(main)
