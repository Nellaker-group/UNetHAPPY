"""
Converts saved model predictions into a tsv that QuPath can read
"""
from pathlib import Path
import pandas as pd

import nucnet.db.eval_runs_interface as db
from nucnet.hdf5.utils import filter_hdf5


def main():
    db.init()

    run_id = 4
    slide_name = "1003608"
    filtered = False

    if not filtered:
        save_path = Path("tsvs") / f"{slide_name}.tsv"
        coords, preds = _get_db_predictions(run_id)
    else:
        min_conf = 1.0
        max_conf = 1.0
        save_path = Path("tsvs") / f"{min_conf}_{max_conf}_{slide_name}.tsv"
        coords, preds = _get_filtered_confidence_predictions(run_id, min_conf, max_conf)

    coord_to_tsv(coords, preds, save_path)


def coord_to_tsv(coords, preds, save_path, nuclei_only=False):
    labels_dict = {None: "NA", 0: "CYT", 1: "FIB", 2: "HOF", 3: "SYN", 4: "VEN"}

    xs = [coord["x"] for coord in coords]
    ys = [coord["y"] for coord in coords]
    if not nuclei_only:
        cell_class = [labels_dict[pred["cell_class"]] for pred in preds]
    else:
        cell_class = ["Nuclei" for _ in coords]

    print(f"saving {len(preds)} cells to tsv")
    df = pd.DataFrame({"x": xs, "y": ys, "class": cell_class})
    df.to_csv(save_path, sep="\t", index=False)


def _get_db_predictions(run_id):
    coords = db.get_predictions(run_id)
    return coords, coords


def _get_filtered_confidence_predictions(run_id, metric_start, metric_end):
    embeddings_dir = (
        (Path(__file__).parent.parent).absolute() / "results" / "embeddings"
    )
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = db.get_embeddings_path(run_id, embeddings_dir)
    embeddings_file = embeddings_dir / embeddings_path

    predictions, _, coords, _, _, _ = filter_hdf5(
        embeddings_file,
        start=0,
        num_points=-1,
        metric_type="confidence",
        metric_start=metric_start,
        metric_end=metric_end,
    )
    print(f"confidence bounds: {metric_start}-{metric_end}")

    return coords, predictions


if __name__ == "__main__":
    main()
