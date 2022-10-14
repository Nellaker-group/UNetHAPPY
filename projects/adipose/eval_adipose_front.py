import time
from typing import Optional

import typer

from happy.organs.organs import get_organ
from happy.utils.utils import get_device
import eval_adipose 
import db.eval_runs_interface as db


# emil changed this to try and make it run
def main(
    project_name: str = typer.Option(...),
    organ_name: str = typer.Option(...),
    seg_model_id: Optional[int] = None,
    run_id: Optional[int] = None,
    slide_id: Optional[int] = None,
    seg_num_workers: int = 10,
    score_threshold: float = 0.8,
    # emil chainged batch to 1
    seg_batch_size: int = 2,
    run_segment_pipeline: bool = True,
    # emil
    # get_cuda_device_num: bool = False,
    get_cuda_device_num: bool = True,
):
    """Runs inference over a WSI for segment detection, cell classification, or both.

    Cell classification alone requires segment detection to have already been
    performed and validated. Will make a new run_id if there are no segment detections,
    otherwise it will pick up an existing run to continue segment detection, start cell
    classification or continue cell classification.

    Predictions are saved to the database with every batch.

    Args:
        project_name: name of the project dir to save results to
        organ_name: name of organ for getting the cells
        nuc_model_id: id of the segment model for inference
        cell_model_id: id of the cell model for inference
        run_id: id of an existing run or of a new run. If none, will auto increment
        slide_id: id of the WSI. Only optional for cell eval.
        nuc_num_workers: number of workers for parallel processing of segment inference
        cell_num_workers: number of workers for parallel processing of cell inference
        score_threshold: segment network confidence cutoff for saving predictions
        max_detections: max segment detections for saving predictions
        nuc_batch_size: batch size for segment inference
        cell_batch_size: batch size for cell inference
        cell_saving: True if you want to save cell predictions to database
        run_segment_pipeline: True if you want to perform segment detection
        run_cell_pipeline: True if you want to perform cell classification
        get_cuda_device_num: if you want the code to choose a gpu
    """
    device = get_device(get_cuda_device_num)
    print("device:")
    print(device)

    # emil the database is the central thing this pipeline writes to
    # Create database connection
    db.init()

    if run_segment_pipeline:
        # Start timer for segment evaluation
        start = time.time()
        # Perform all segment evaluation
        run_id = segment_eval_pipeline(
            seg_model_id,
            slide_id,
            run_id,
            seg_num_workers,
            seg_batch_size,
            score_threshold,
            device,
        )
        end = time.time()
        print(f"Segment evaluation time: {(end - start):.3f}")

# emil this is the main method that is being run
def segment_eval_pipeline(
    model_id,
    slide_id,
    run_id,
    num_workers,
    batch_size,
    score_threshold,
    device,
):
    # emil this is inside the eval_adipose.py
    # Load model weights and push to device
    model = eval_adipose.setup_model(model_id, device, 1, 3, 1)
    
    # Load dataset and dataloader
    # emil hardcoded overlap of 0  
    #dataloader, pred_saver = eval_adipose.setup_data(
    #    slide_id, run_id, model_id, batch_size, overlap=200, num_workers=num_workers
    #)
    dataloader, pred_saver = eval_adipose.setup_data(
        slide_id, run_id, model_id, batch_size, overlap=512, num_workers=num_workers
    )
    # Predict segment
    eval_adipose.run_seg_eval(
        dataloader, model, pred_saver, device, score_threshold
    )
    eval_adipose.clean_up(pred_saver)
    return pred_saver.id

if __name__ == "__main__":
    typer.run(main)
