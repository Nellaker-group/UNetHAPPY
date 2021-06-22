import umap
import umap.plot
import pandas as pd
import numpy as np
from bokeh.plotting import output_file


def plot_interactive(
    plot_name, slide_name, organ, predictions, confidence, coords, mapper
):
    output_file(plot_name, title=f"UMAP Embeddings of Slide {slide_name}")

    label_colours = {cell.id: cell.colour for cell in organ.cells}
    label_ids = {cell.id: cell.label for cell in organ.cells}

    df = pd.DataFrame(
        {
            "pred": predictions,
            "confidence": confidence,
            "x_": coords[:, 0],
            "y_": coords[:, 1],
        }
    )
    df["pred"] = df.pred.map(label_ids)

    return umap.plot.interactive(
        mapper,
        labels=predictions,
        color_key=label_colours,
        interactive_text_search=True,
        hover_data=df,
        point_size=2,
    )


def plot_umap(organ, predictions, mapper):
    colours_dict = {cell.label: cell.colour for cell in organ.cells}
    predictions_labelled = np.array([organ.cells[pred].label for pred in predictions])
    return umap.plot.points(mapper, labels=predictions_labelled, color_key=colours_dict)
