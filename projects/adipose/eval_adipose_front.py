import time
from typing import Optional

import typer

from happy.organs.organs import get_organ
from happy.utils.utils import get_device
import eval_adipose 
import db.eval_runs_interface as db


def main(
    project_name: str = typer.Option(...),
    organ_name: str = typer.Option(...),
    seg_model_id: Optional[int] = None,
    run_id: Optional[int] = None,
    slide_id: Optional[int] = None,
    # emil added this for parallel databases - as concurence issues might happen when running it in parallel
    database_id: Optional[int] = None,
    seg_num_workers: int = 20,
    score_threshold: float = 0.8,
    seg_batch_size: int = 2,
    # emil added the pixel_size option
    pixel_size: float = 0.2500,
    run_segment_pipeline: bool = True,
    get_cuda_device_num: bool = False,
    write_geojson: bool = False,
):
    """Runs inference over a WSI where it does semantic segmentation.

    Predictions, in the shape of polygons, are saved to the database with every batch.

    Args:
        project_name: name of the project dir to save results to
        organ_name: name of organ for getting the cells
        seg_model_id: id of the segment model for inference
        run_id: id of an existing run or of a new run. If none, will auto increment
        slide_id: id of the WSI. Only optional for cell eval.
        database_id: id of the database or .db file being written to.
        seg_num_workers: number of workers for parallel processing of segment inference
        score_threshold: segment network confidence cutoff for saving predictions
        max_detections: max segment detections for saving predictions
        nuc_batch_size: batch size for segment inference
        seg_batch_size: batch size for segmentation
        run_segment_pipeline: True if you want to run the segmentation pipeline
        get_cuda_device_num: if you want the code to choose a gpu
        write_geojson: if you want to write a geojson file of the output
    """
    device = get_device(get_cuda_device_num)
    print("device:")
    print(device)

    # Create database connection
    if database_id != None:
        db.init("Batch_"+str(database_id)+".db")
    else:
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
            write_geojson,
            pixel_size,
        )
        end = time.time()
        print(f"Segment evaluation time: {(end - start):.3f}")


def segment_eval_pipeline(
    model_id,
    slide_id,
    run_id,
    num_workers,
    batch_size,
    score_threshold,
    device,
    write_geojson,
    pixel_size,
):

    # Load model weights and push to device
    model = eval_adipose.setup_model(model_id, device, 1, 3, 1)
    
    # Load dataset and dataloader
    dataloader, pred_saver = eval_adipose.setup_data(
        slide_id, run_id, model_id, batch_size, overlap=256, num_workers=num_workers, pixel_size=pixel_size
    )
    # Predict segment
    eval_adipose.run_seg_eval(
        dataloader, model, pred_saver, device, write_geojson, score_threshold
    )
    eval_adipose.clean_up(pred_saver)
    return pred_saver.id

if __name__ == "__main__":
    typer.run(main)
