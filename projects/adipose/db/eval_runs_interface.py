from projects.adipose.db.eval_runs import (
    EvalRun,
    Prediction,
    UnvalidatedPrediction,
    MergedPrediction,
)
from projects.adipose.db.base import database, init_db
from happy.db.eval_runs_interface import *


def init(db_name="main.db"):
    db_path = Path(__file__).parent.absolute() / db_name
    init_db(db_path)


def mark_seg_as_done(run_id):
    eval_run = EvalRun.get_by_id(run_id)
    eval_run.seg_done = True
    eval_run.save()


def save_pred_workings(run_id, coords, latest_poly_id):
    # Emil put the db.atomic as this is apparently faster according to the peewee documentation
    # coords is a list of tuples with each one holding the poly_id, point_id, X and Y coordinate for a point (point_id is needed as the same polygon might have identical points due to rounding off to ints)
    fields = [
        UnvalidatedPrediction.run,
        UnvalidatedPrediction.poly_id,
        UnvalidatedPrediction.point_id,
        UnvalidatedPrediction.X,
        UnvalidatedPrediction.Y,
    ]
    data = [
        (run_id, coord[0] + latest_poly_id, coord[1], coord[2], coord[3])
        for coord in coords
    ]

    with database.atomic():
        for batch in chunked(data, 10):
            UnvalidatedPrediction.insert_many(batch, fields=fields).execute()


def get_all_unvalidated_seg_preds(run_id):
    preds = UnvalidatedPrediction.select(
        UnvalidatedPrediction.poly_id,
        UnvalidatedPrediction.point_id,
        UnvalidatedPrediction.X,
        UnvalidatedPrediction,
    ).where(UnvalidatedPrediction.run == run_id)
    preds2 = list(preds)
    listie = [(dic.poly_id, dic.point_id, dic.X, dic.Y) for dic in preds2]
    return listie


def get_all_validated_seg_preds(run_id):
    preds = Prediction.select(
        Prediction.poly_id, Prediction.point_id, Prediction.X, Prediction.Y
    ).where(Prediction.run == run_id)
    preds2 = list(preds)
    listie = [(dic.poly_id, dic.point_id, dic.X, dic.Y) for dic in preds2]
    return listie


def get_all_merged_seg_preds(run_id1, run_id2):
    preds = MergedPrediction.select(
        MergedPrediction.poly_id,
        MergedPrediction.point_id,
        MergedPrediction.X,
        MergedPrediction.Y,
    ).where((MergedPrediction.run1 == run_id1) & (MergedPrediction.run2 == run_id2))
    preds2 = list(preds)
    listie = [(dic.poly_id, dic.point_id, dic.X, dic.Y) for dic in preds2]
    return listie


def save_validated_polygons(run_id, valid_coords):
    # valid_coords looks like this [ ("poly_id", "point_id", "X", "Y"), ... ]
    print(f"marking {len(valid_coords)} polygons as valid")
    fields = [
        Prediction.run,
        Prediction.poly_id,
        Prediction.point_id,
        Prediction.X,
        Prediction.Y,
    ]
    data = [(run_id, coord[0], coord[1], coord[2], coord[3]) for coord in valid_coords]
    with database.atomic():
        for batch in chunked(data, 10):
            Prediction.insert_many(batch, fields=fields).execute()


def validate_merged_workings(run_id1, run_id2, merged_coords):
    # merged_coords looks like this [ ("poly_id", "point_id", "X", "Y"), ... ]
    print(f"marking {len(merged_coords)} polygons as merged")
    fields = [
        MergedPrediction.run1,
        MergedPrediction.run2,
        MergedPrediction.poly_id,
        MergedPrediction.point_id,
        MergedPrediction.X,
        MergedPrediction.Y,
    ]
    data = [
        (run_id1, run_id2, coord[0], coord[1], coord[2], coord[3])
        for coord in merged_coords
    ]
    with database.atomic():
        for batch in chunked(data, 10):
            MergedPrediction.insert_many(batch, fields=fields).execute()
