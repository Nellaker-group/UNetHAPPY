import nucnet.db.eval_runs_interface as db

# gets nuclei locations within a specified box
def get_nuc_locs(run_id, file, min_x, min_y, width, height):
    width = int(width * file.rescale_ratio)
    height = int(height * file.rescale_ratio)
    max_x = min_x + width
    max_y = min_y + height

    nuc_preds = db.get_nuclei_in_range(run_id, min_x, min_y, max_x, max_y)
    if not nuc_preds:
        raise ValueError(
            "no nuclei detections found within that range. Ending execution."
        )
    return nuc_preds
