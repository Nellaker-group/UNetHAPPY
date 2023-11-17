import sys
import os

# to be able to import packages
sys.path.append(os.getcwd())

from db.base import database as db
from db.eval_runs import EvalRun, TileState, Prediction, UnvalidatedPrediction
from db.eval_runs_interface import init

init(db_name="baseV2.db")
db.drop_tables([EvalRun, TileState, Prediction, UnvalidatedPrediction])
