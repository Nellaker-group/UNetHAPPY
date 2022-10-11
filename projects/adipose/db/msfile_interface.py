from pathlib import Path

from db.eval_runs import EvalRun
from microscopefile.microscopefile import MicroscopeFile
from microscopefile.reader import Reader


# returns an ms_file object with values from db if supplied with run_id
def get_msfile(
    slide_id=None,
    run_id=None,
    seg_model_id=None,
    tile_width=1024,
    tile_height=1024,
    # emil changed pixel size to 0.2500 and overlap to 0
    pixel_size=0.2500,
    overlap=0,
    subsect_x=None,
    subsect_y=None,
    subsect_w=None,
    subsect_h=None,
):
    if not run_id:
        # Creates a new run with a new run_id
        run = EvalRun.create(
            seg_model=seg_model_id,
            slide=slide_id,
            tile_width=tile_width,
            tile_height=tile_height,
            pixel_size=pixel_size,
            overlap=overlap,
            subsect_x=subsect_x,
            subsect_y=subsect_y,
            subsect_w=subsect_w,
            subsect_h=subsect_h,
        )
        print(f"no run id given, making new run with id {run.id}")
    else:
        run = EvalRun.get_or_none(EvalRun.id == run_id)
        if not run:
            # Creates a new run with a new run_id (supplied run_id wasn't valid)
            run = EvalRun.create(
                seg_model=seg_model_id,
                slide=slide_id,
                tile_width=tile_width,
                tile_height=tile_height,
                pixel_size=pixel_size,
                overlap=overlap,
                subsect_x=subsect_x,
                subsect_y=subsect_y,
                subsect_w=subsect_w,
                subsect_h=subsect_h,
            )
            print(f"no run with id {run_id}, making new run with id {run.id}")
        else:
            # Uses run_id to get and continue existing run
            if seg_model_id and run.seg_model is None:
                run.seg_model = seg_model_id
                run.save()
            print("using existing microscopefile")

    return _init_msfile(run)


# returns an ms_file object from existing eval run
def get_msfile_by_run(run_id):
    run = EvalRun.get_by_id(run_id)
    return _init_msfile(run)


def _init_msfile(run):
    full_slide_path = str(Path(run.slide.lab.slides_dir) / run.slide.slide_name)
    reader = Reader.new(full_slide_path, run.slide.lvl_x)
    ## emil 
    print("run.pixel_size:")
    print(run.pixel_size)
    print("run.slide.pixel_size:")
    print(run.slide.pixel_size)
    return MicroscopeFile(
        run.id,
        reader,
        full_slide_path,
        run.tile_width,
        run.tile_height,
        run.pixel_size,
        run.slide.pixel_size,
        run.overlap,
        run.subsect_x,
        run.subsect_y,
        run.subsect_h,
        run.subsect_w,
        run.segs_done,
    )
