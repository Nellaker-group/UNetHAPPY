import time

import typer
from torch import cuda

from happy.utils.utils import set_gpu_device
from happy.eval import nuclei_eval, cell_eval
import happy.db.eval_runs_interface as db


def main(
    project_name: str = "placenta",
    nuc_model_id: int = -1,
    cell_model_id: int = -1,
    run_id: int = -1,
    slide_id: int = -1,
    num_cells: int = 5,
    nuc_num_workers: int = 20,
    cell_num_workers: int = 16,
    score_threshold: float = 0.3,
    max_detections: int = 150,
    cell_batch_size: int = 800,
    cell_saving: bool = True,
    run_nuclei_pipeline: bool = True,
    run_cell_pipeline: bool = True,
):
    """Runs inference over a WSI for nuclei detection, cell classification, or both.

    Cell classification alone requires nuclei detection to have already been
    performed and validated. Will make a new run_id if there are no nuclei detections,
    otherwise it will pick up an existing run to continue nuclei detection, start cell
    classification or continue cell classification.

    Predictions are saved to the database with every batch.

    Args:
        nuc_model_id: id of the nuclei model for inference
        cell_model_id: id of the cell model for inference
        run_id: id of an existing run if continuing or of a new run
        slide_id: id of the WSI
        num_cells: number of cell classes (default 5 for placenta)
        nuc_num_workers: number of workers for parallel processing of nuclei inference
        cell_num_workers: number of workers for parallel processing of cell inference
        score_threshold: nuclei network confidence cutoff for saving predictions
        max_detections: max nuclei detections for saving predictions
        cell_batch_size: batch size for cell inference
        cell_saving: True if you want to save cell predictions to database
        run_nuclei_pipeline: True if you want to perform nuclei detection
        run_cell_pipeline: True if you want to perform cell classification
    """
    if cuda.is_available():
        set_gpu_device()
        device = "cuda"
    else:
        device = "cpu"

    # Create database connection
    db.init()

    if run_nuclei_pipeline:
        # Start timer for nuclei evaluation
        start = time.time()
        # Perform all nuclei evaluation
        nuclei_eval_pipeline(
            nuc_model_id,
            slide_id,
            run_id,
            nuc_num_workers,
            score_threshold,
            max_detections,
            device,
        )
        end = time.time()
        print(f"Nuclei evaluation time: {(end - start):.3f}")

    if run_cell_pipeline:
        # Start timer for cell evaluation
        start = time.time()
        # Perform all nuclei evaluation
        cell_eval_pipeline(
            project_name,
            cell_model_id,
            run_id,
            num_cells,
            cell_batch_size,
            cell_num_workers,
            device,
            cell_saving=cell_saving,
        )
        end = time.time()
        print(f"Cell evaluation time: {(end - start):.3f}")


def nuclei_eval_pipeline(
    model_id, slide_id, run_id, num_workers, score_threshold, max_detections, device
):
    # Load model weights and push to device
    model = nuclei_eval.setup_model(model_id)
    # Load dataset and dataloader
    dataloader, pred_saver = nuclei_eval.setup_data(
        slide_id, run_id, model_id, overlap=200, num_workers=num_workers
    )
    # Predict nuclei
    nuclei_eval.run_nuclei_eval(
        dataloader, model, pred_saver, score_threshold, max_detections
    )
    nuclei_eval.clean_up(pred_saver)


def cell_eval_pipeline(
    project_name,
    model_id,
    run_id,
    out_features,
    batch_size,
    num_workers,
    device,
    cell_saving=True,
):
    # Load model weights and push to device
    model, model_architecture = cell_eval.setup_model(model_id, out_features, device)
    # Load dataset and dataloader
    dataloader, pred_saver = cell_eval.setup_data(
        run_id,
        model_id,
        model_architecture,
        batch_size=batch_size,
        num_workers=num_workers,
        cell_saving=cell_saving,
    )
    # Setup or get path to embeddings hdf5 save location
    embeddings_path = cell_eval.setup_embedding_saving(
        project_name, run_id, cell_saving
    )
    # Predict cell classes
    cell_eval.run_cell_eval(
        dataloader, model, pred_saver, embeddings_path, device, cell_saving
    )
    cell_eval.clean_up(pred_saver)


if __name__ == "__main__":
    typer.run(main)
