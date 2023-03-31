from pathlib import Path
from collections import defaultdict

from peewee import Tuple, ValuesList, EnclosedNodeList, chunked

from db.slides import Slide
from db.eval_runs import EvalRun, TileState, Prediction, UnvalidatedPrediction, MergedPrediction
from db.models_training import Model
from db.base import database, init_db


def init(db_name = "main.db"):
    db_path = Path(__file__).parent.absolute() / db_name
    print("db_path:")
    print(db_path)
    init_db(db_path)


# returns the path to a slide
def get_slide_path_by_id(slide_id):
    slide = Slide.get_by_id(slide_id)
    return Path(slide.lab.slides_dir) / slide.slide_name


# returns a slide
def get_slide_by_id(slide_id):
    return Slide.get_by_id(slide_id)


# returns the path to model_weights
def get_model_weights_by_id(model_id):
    ## emil
    print("model_id:")
    print(model_id)
    model = Model.get_by_id(model_id)
    return model.architecture, model.path


# returns an eval run
def get_eval_run_by_id(run_id):
    eval_run = EvalRun.get_by_id(run_id)
    return eval_run


# TODO: change this to always return a Path object
def get_embeddings_path(run_id, embeddings_dir=None):
    eval_run = EvalRun.get_by_id(run_id)
    if not eval_run.embeddings_path:
        slide = eval_run.slide
        lab_id = slide.lab.id
        slide_name = slide.slide_name

        embeddings_path = Path(f"lab_{lab_id}") / f"slide_{slide_name}"
        (Path(embeddings_dir) / embeddings_path).mkdir(parents=True, exist_ok=True)
        path_with_file = embeddings_path / f"run_{run_id}.hdf5"

        eval_run.embeddings_path = path_with_file
        eval_run.save()
        return str(path_with_file)
    else:
        return eval_run.embeddings_path


# Updates temporary run tile state table with a new tiles run state
def save_new_tile_state(run_id, tile_xy_list):
    fields = [TileState.run, TileState.tile_index, TileState.tile_x, TileState.tile_y]

    xs = [x[0] for x in tile_xy_list]
    ys = [y[1] for y in tile_xy_list]

    data = [(run_id, i, xs[i], ys[i]) for i in range(len(tile_xy_list))]

    with database.atomic():
        for batch in chunked(data, 8000):
            TileState.insert_many(batch, fields=fields).execute()


# Returns False if state is None, otherwise True
def run_state_exists(run_id):
    state = TileState.get_or_none(TileState.run == run_id)
    return True if state else False


def get_run_state(run_id):
    tile_states = (
        TileState.select(TileState.tile_x, TileState.tile_y)
        .where(TileState.run == run_id)
        .tuples()
    )
    return tile_states


def get_remaining_tiles(run_id):
    with database.atomic():
        tile_coords = (
            TileState.select(TileState.tile_index, TileState.tile_x, TileState.tile_y)
            .where((TileState.run == run_id) & (TileState.done == False))
            .dicts()
        )
    return tile_coords


def mark_finished_tiles(run_id, tile_indexes):
    with database.atomic():
        query = TileState.update(done=True).where(
            (TileState.run == run_id) & (TileState.tile_index << tile_indexes)
        )
        query.execute()


def mark_seg_as_done(run_id):
    eval_run = EvalRun.get_by_id(run_id)
    eval_run.seg_done = True
    eval_run.save()


def save_pred_workings(run_id, coords, latest_poly_id):
    # Emil put the db.atomic as this is apparently faster according to the peewee documentation
    # coords is a list of tuples with each one holding the poly_id, point_id, X and Y coordinate for a point (point_id is needed as the same polygon might have identical points due to rounding off to ints)
    fields = [UnvalidatedPrediction.run, UnvalidatedPrediction.poly_id, UnvalidatedPrediction.point_id, UnvalidatedPrediction.X, UnvalidatedPrediction.Y]
    data = [(run_id, coord[0]+latest_poly_id, coord[1], coord[2], coord[3]) for coord in coords]

    with database.atomic():
        for batch in chunked(data, 10):
            UnvalidatedPrediction.insert_many(batch, fields=fields).execute()


def get_all_unvalidated_seg_preds(run_id):
    preds = (
        UnvalidatedPrediction.select(UnvalidatedPrediction.poly_id, UnvalidatedPrediction.point_id, UnvalidatedPrediction.X, UnvalidatedPrediction)
        .where(UnvalidatedPrediction.run == run_id)
    )
    preds2 = list(preds)
    listie = [(dic.poly_id, dic.point_id, dic.X, dic.Y) for dic in preds2]
    return listie


def get_all_validated_seg_preds(run_id):
    preds = (
        Prediction.select(Prediction.poly_id, Prediction.point_id, Prediction.X, Prediction.Y)
        .where(Prediction.run == run_id)
    )
    preds2 = list(preds)
    listie = [(dic.poly_id, dic.point_id, dic.X, dic.Y) for dic in preds2]
    return listie


def get_all_merged_seg_preds(run_id1, run_id2):
    preds = (
        MergedPrediction.select(MergedPrediction.poly_id, MergedPrediction.point_id, MergedPrediction.X, MergedPrediction.Y)
        .where((MergedPrediction.run1) == run_id1 & (MergedPrediction.run2 == run_id2))
    )
    preds2 = list(preds)
    listie = [(dic.poly_id, dic.point_id, dic.X, dic.Y) for dic in preds2]
    return listie

# right now this does the same as save_pred_workings() - however that should probably change
def commit_pred_workings(run_id, coords):
    with database.atomic():
        for coord in coords:
            Prediction.create(**coord)
    Prediction.execute()


def validate_pred_workings(run_id, valid_coords):
    # valid_coords looks like this [ ("poly_id", "point_id", "X", "Y"), ... ]
    print(f"marking {len(valid_coords)} polygons as valid")
    fields = [Prediction.run, Prediction.poly_id, Prediction.point_id, Prediction.X, Prediction.Y]
    data = [(run_id, coord[0], coord[1], coord[2],  coord[3]) for coord in valid_coords]
    with database.atomic():
        for batch in chunked(data, 10):
            Prediction.insert_many(batch, fields=fields).execute()


def validate_merged_workings(run_id1, run_id2, merged_coords):
    # merged_coords looks like this [ ("poly_id", "point_id", "X", "Y"), ... ]
    print(f"marking {len(merged_coords)} polygons as merged")
    fields = [MergedPrediction.run1, MergedPrediction.run2, MergedPrediction.poly_id, MergedPrediction.point_id, MergedPrediction.X, MergedPrediction.Y]
    data = [(run_id1, run_id2, coord[0], coord[1], coord[2],  coord[3]) for coord in merged_coords]
    with database.atomic():
        for batch in chunked(data, 10):
            MergedPrediction.insert_many(batch, fields=fields).execute()


def get_num_remaining_tiles(run_id):
    return (
        TileState.select()
        .where((TileState.run == run_id) & (TileState.done == False))
        .count()
    )


def get_num_remaining_cells(run_id):
    return (
        Prediction.select()
        .where((Prediction.run == run_id) & (Prediction.cell_class.is_null(True)))
        .count()
    )


def get_total_num_nuclei(run_id):
    return Prediction.select().where(Prediction.run == run_id).count()


def get_predictions(run_id):
    return (
        Prediction.select(Prediction.x, Prediction.y, Prediction.cell_class)
        .where(Prediction.run == run_id)
        .dicts()
    )


def get_all_prediction_coordinates(run_id):
    return (
        Prediction.select(Prediction.x, Prediction.y)
        .where(Prediction.run == run_id)
        .dicts()
    )


def get_slide_name(run_id):    
    eval = EvalRun.get_by_id(run_id)
    slide = Slide.get_by_id(eval.slide_id)
    return(slide.slide_name)
