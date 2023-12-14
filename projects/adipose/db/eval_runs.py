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

from projects.adipose.db.base import BaseModel
from projects.adipose.db.slides import Slide
from projects.adipose.db.models_training import Model


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
    seg_done = BooleanField(default=False)


class Prediction(BaseModel):
    run = ForeignKeyField(EvalRun, backref="prediction_polys")
    poly_id = IntegerField()
    point_id = IntegerField()
    X = IntegerField()
    Y = IntegerField()
    poly_class = IntegerField(null=True)

    class Meta:
        primary_key = CompositeKey("run", "poly_id", "point_id", "X", "Y")


class TileState(BaseModel):
    run = ForeignKeyField(EvalRun, backref="tile_states")
    tile_index = IntegerField()
    tile_x = IntegerField()
    tile_y = IntegerField()
    done = BooleanField(default=False)

    class Meta:
        primary_key = CompositeKey("run", "tile_index")


class UnvalidatedPrediction(BaseModel):
    run = ForeignKeyField(EvalRun, backref="unvalidated_prediction_polys")
    poly_id = IntegerField()
    point_id = IntegerField()
    X = IntegerField()
    Y = IntegerField()
    poly_class = IntegerField(null=True)

    class Meta:
        primary_key = CompositeKey("run", "poly_id", "point_id", "X", "Y")


class MergedPrediction(BaseModel):
    run1 = ForeignKeyField(EvalRun, backref="merged_prediction_polys1")
    run2 = ForeignKeyField(EvalRun, backref="merged_prediction_polys2")
    poly_id = IntegerField()
    point_id = IntegerField()
    X = IntegerField()
    Y = IntegerField()
    poly_class = IntegerField(null=True)

    class Meta:
        primary_key = CompositeKey("run1", "run2", "poly_id", "point_id", "X", "Y")
