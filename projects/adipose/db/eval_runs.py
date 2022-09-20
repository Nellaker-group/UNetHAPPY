from datetime import datetime

from peewee import (
    IntegerField,
    FloatField,
    ForeignKeyField,
    BooleanField,
    DateTimeField,
    CompositeKey,
    TextField,
)

from db.base import BaseModel
from db.slides import Slide
from db.models_training import Model


class EvalRun(BaseModel):
    timestamp = DateTimeField(formats="%Y-%m-%d %H:%M", default=datetime.utcnow())
    seg_model = ForeignKeyField(Model, backref="eval_runs")
    slide = ForeignKeyField(Slide, backref="eval_runs")
    tile_width = IntegerField(default=1024)
    tile_height = IntegerField(default=1024)
    # emil changed pixel size to 0.2500 and overlap to 0
    pixel_size = FloatField(default=0.2500)
    overlap = IntegerField(default=0)
    subsect_x = IntegerField(null=True)
    subsect_y = IntegerField(null=True)
    subsect_w = IntegerField(null=True)
    subsect_h = IntegerField(null=True)
    embeddings_path = TextField(null=True)
    segs_done = BooleanField(default=False)


class Prediction(BaseModel):
    run = ForeignKeyField(EvalRun, backref="predictions")
    x = IntegerField()
    y = IntegerField()
    cell_class = IntegerField(null=True)

    class Meta:
        primary_key = CompositeKey("run", "x", "y")


class PredictionString(BaseModel):
    run = ForeignKeyField(EvalRun, backref="predictions")
    polyXY = TextField()

    class Meta:
        primary_key = CompositeKey("run", "polyXY")


class TileState(BaseModel):
    run = ForeignKeyField(EvalRun, backref="tile_states")
    tile_index = IntegerField()
    tile_x = IntegerField()
    tile_y = IntegerField()
    done = BooleanField(default=False)

    class Meta:
        primary_key = CompositeKey("run", "tile_index")


class UnvalidatedPrediction(BaseModel):
    run = ForeignKeyField(EvalRun, backref="unvalidated_predictions")
    x = IntegerField()
    y = IntegerField()
    is_valid = BooleanField(default=False)

    class Meta:
        primary_key = CompositeKey("run", "x", "y")


class UnvalidatedPredictionString(BaseModel):
    run = ForeignKeyField(EvalRun, backref="unvalidated_predictions")
    polyXY = TextField()
    is_valid = BooleanField(default=False)

    class Meta:
        primary_key = CompositeKey("run", "polyXY")
