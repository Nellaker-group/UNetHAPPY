from pathlib import Path
from collections import defaultdict

from peewee import Tuple, ValuesList, EnclosedNodeList, chunked

from db.slides import Slide
from db.eval_runs import EvalRun, TileState, PredictionString, UnvalidatedPredictionString
from db.models_training import Model
from db.base import database, init_db


def init():
    db_name = "main.db"
    #emil
    #db_path = Path(__file__).parent.absolute() / db_name
    db_path="/well/lindgren/users/swf744/git/HAPPY/projects/adipose/db/main.db"
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


# emil function for converting string into list of tuples - so decoding
def stringListTuple2coordinates(string):
    splitString = string.replace("),",")|").split("|")
    returnList = []
    for ele in splitString:
        tmp = ele.replace("[","").replace("]","").replace("(","").replace(")","")
        if tmp != "":
            x, y = float(tmp.split(",")[0]), float(tmp.split(",")[1])
            returnList.append((x,y))
    return(returnList)


# I store my polygons as strings of a list of tuples with the coordinates
def save_pred_workings(run_id, coords, tile_index):
    # I put the db.atomic as this is apparently faster according to the peewee documentation
    # coords is a list of dicts with the keys according to the columns of the database
    # coords looks like this [ (items["polyXY"], items["polyID"]), ... ]
    fields = [UnvalidatedPredictionString.run, UnvalidatedPredictionString.polyID, UnvalidatedPredictionString.polyXY, UnvalidatedPredictionString.tile_index]    
    data = [(run_id, coords[i]["polyID"], coords[i]["polyXY"], tile_index) for i in range(len(coords))]
    with database.atomic():
        for batch in chunked(data, 10):
            UnvalidatedPredictionString.insert_many(batch, fields=fields).execute()

def get_all_unvalidated_seg_preds(run_id):
    preds = (
        UnvalidatedPredictionString.select(UnvalidatedPredictionString.polyXY)
        .where(UnvalidatedPredictionString.run == run_id)
    )
    preds2 = list(preds)
    listie = [dic.polyXY for dic in preds2] 
    return listie


# right now this does the same as save_pred_workings() - however that should probably change
def commit_pred_workings(run_id, coords):
    with database.atomic():
        for coord in coords:
            PredictionString.create(**coord)
    PredictionString.execute()


# this should probably be utilised at some point
def validate_pred_workings(run_id, valid_coords):
    # valid_coords looks like this [ (items["polyXY"], items["polyID"]), ... ]
    print(f"marking {len(valid_coords)} polygons as valid ")
    fields = [PredictionString.run, PredictionString.polyID, PredictionString.polyXY]
    data = [(run_id, valid_coords[i]["polyID"], valid_coords[i]["polyXY"]) for i in range(len(valid_coords))]
    with database.atomic():
        for batch in chunked(data, 10):
            PredictionString.insert_many(batch, fields=fields).execute()



def commit_pred_workingsOLD(run_id):
    source = (
        UnvalidatedPrediction.select(
            UnvalidatedPrediction.run, UnvalidatedPrediction.x, UnvalidatedPrediction.y
        )
        .where(
            (UnvalidatedPrediction.run == run_id)
            & (UnvalidatedPrediction.is_valid == True)
        )
        .order_by(UnvalidatedPrediction.x, UnvalidatedPrediction.y.asc())
    )

    rows = Prediction.insert_from(
        source, fields=[Prediction.run, Prediction.x, Prediction.y]
    ).execute()
    print(f"added {rows} nuclei predictions to Predictions table for eval run {run_id}")


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


def get_nuclei_in_range(run_id, min_x, min_y, max_x, max_y):
    # Get predictions within specified range
    return (
        Prediction.select(Prediction.x, Prediction.y, Prediction.cell_class)
        .where(
            (Prediction.run == run_id)
            & (Tuple(Prediction.x, Prediction.y) >= (min_x, min_y))
            & (Tuple(Prediction.x, Prediction.y) <= (max_x, max_y))
        )
        .tuples()
    )
