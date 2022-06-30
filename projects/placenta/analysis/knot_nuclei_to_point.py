import typer

from happy.organs.organs import get_organ
from happy.utils.utils import get_project_dir
import happy.db.eval_runs_interface as db
from happy.hdf5.utils import (
    get_datasets_in_patch,
    get_embeddings_file,
)


def main(
    run_id: int = typer.Option(...),
    project_name: str = "placenta",
    x_min: int = 0,
    y_min: int = 0,
    width: int = -1,
    height: int = -1,
):
    # Create database connection
    db.init()

    organ = get_organ("placenta")
    project_dir = get_project_dir(project_name)

    # Get path to embeddings hdf5 files
    embeddings_path = get_embeddings_file(project_name, run_id)
    # Get hdf5 datasets contained in specified box/patch of WSI
    predictions, embeddings, coords, confidence = get_datasets_in_patch(
        embeddings_path, x_min, y_min, width, height
    )
    print(f"Data loaded with {len(predictions)} nodes")





if __name__ == "__main__":
    main()
