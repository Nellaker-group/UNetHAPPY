import torch
import pandas as pd

from happy.utils.utils import get_project_dir
from happy.graph.graph_creation.get_and_process import get_hdf5_data
from happy.graph.graph_creation.create_graph import setup_cell_tissue_graph


def main():
    project_dir = get_project_dir("placenta")
    lesion_dirs = project_dir / "annotations" / "lesion" / "single"
    save_dir = project_dir / "datasets" / "lesion" / "single"

    for split in ["train_lesion", "val_lesion", "test_lesion"]:
        df = pd.read_csv(lesion_dirs / f"{split}.csv")

        for index, row in df.iterrows():
            run_id = row["run_id"]

            hdf5_data = get_hdf5_data("placenta", run_id, 0, 0, -1, -1, tissue=True)
            data = setup_cell_tissue_graph(hdf5_data, k=8, graph_method="intersection")
            torch.save(data, save_dir / f"run_{run_id}.pt")


if __name__ == "__main__":
    main()