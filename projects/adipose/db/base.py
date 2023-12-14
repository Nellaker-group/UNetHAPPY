from peewee import SqliteDatabase, Model

database = SqliteDatabase(None)


def init_db(db_name):
    # Local imports need to create all the tables and avoid circular import of BaseModel
    from projects.adipose.db.slides import Slide, Patient, Lab
    from projects.adipose.db.eval_runs import (
        EvalRun,
        Prediction,
        TileState,
        UnvalidatedPrediction,
        MergedPrediction,
    )
    from projects.adipose.db.models_training import Model, TrainRun
    from projects.adipose.db.tiles import (
        TrainTile,
        Annotation,
        Feature,
        TrainTileRun,
        TileFeature,
    )

    database.init(
        db_name,
        pragmas={
            "foreign_keys": 1,
            "journal_mode": "wal",
            "cache_size": 10000,
            "synchronous": 1,
        },
    )
    database.connect()
    database.create_tables(
        [
            Slide,
            Patient,
            Lab,
            EvalRun,
            Prediction,
            UnvalidatedPrediction,
            MergedPrediction,
            TileState,
            Model,
            TrainRun,
            TrainTile,
            Annotation,
            Feature,
            TrainTileRun,
            TileFeature,
        ]
    )


class BaseModel(Model):
    class Meta:
        database = database
