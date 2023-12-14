from projects.adipose.db.base import database as db
from projects.adipose.db.eval_runs import EvalRun, TileState, Prediction, UnvalidatedPrediction
from projects.adipose.db.eval_runs_interface import init

init(db_name="baseV2.db")
db.drop_tables([EvalRun, TileState, Prediction, UnvalidatedPrediction])
