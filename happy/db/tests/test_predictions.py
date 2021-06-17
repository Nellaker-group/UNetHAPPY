import pathlib
import os

import pytest
import numpy as np

from happy.db.base import init_db
from happy.db.eval_runs_interface import run_state_exists, save_pred_workings, \
    mark_nuclei_as_done, mark_finished_tiles, commit_pred_workings, save_cells, \
    validate_pred_workings, get_remaining_cells
from happy.db.tests.utils import setup_tile_state, setup_eval_run
from happy.db.eval_runs import EvalRun, UnvalidatedPrediction, Prediction


@pytest.fixture()
def db():
    db_name = f"pytest_temp_{__name__}.db"
    db_path = pathlib.Path(__file__).parent.absolute() / db_name
    init_db(db_path)
    yield
    os.remove(db_path)


@pytest.fixture()
def tile_state():
    setup_eval_run()
    setup_tile_state()
    yield


def test_unclustered_predictions(db, tile_state):
    eval_run = EvalRun.get_by_id(1)
    run_id = eval_run.id
    assert eval_run.id == 1
    assert run_state_exists(1)

    # save nuclei and mark all tile states
    coords = [(1, 2), (3, 4), (5, 6)]
    save_pred_workings(run_id, coords)
    mark_finished_tiles(run_id, [0, 1, 2, 3])
    mark_nuclei_as_done(run_id)

    nuclei_predictions = UnvalidatedPrediction.select()

    assert nuclei_predictions[0].x == 1
    assert nuclei_predictions[0].y == 2
    assert nuclei_predictions[0].is_valid == False

    assert nuclei_predictions[1].x == 3
    assert nuclei_predictions[1].y == 4
    assert nuclei_predictions[1].is_valid == False

    assert nuclei_predictions[2].x == 5
    assert nuclei_predictions[2].y == 6
    assert nuclei_predictions[2].is_valid == False

    eval_run = EvalRun.get_by_id(1)
    assert eval_run.nucs_done == True


def test_save_nuclei_predictions(db, tile_state):
    eval_run = EvalRun.get_by_id(1)
    run_id = eval_run.id
    assert eval_run.id == 1
    assert run_state_exists(1)

    # save nuclei and mark all tile states
    coords = [(1, 2), (3, 4), (5, 6), (1, 2)]
    more_coords = [(1, 2)]
    save_pred_workings(run_id, coords)
    save_pred_workings(run_id, more_coords)
    mark_finished_tiles(run_id, [0, 1, 2, 3])
    mark_nuclei_as_done(run_id)

    coords = [(1, 2), (3, 4)]
    validate_pred_workings(run_id, coords)
    commit_pred_workings(run_id)

    nuclei_predictions = Prediction.select()

    assert nuclei_predictions[0].x == 1
    assert nuclei_predictions[0].y == 2

    assert nuclei_predictions[1].x == 3
    assert nuclei_predictions[1].y == 4

    assert len(nuclei_predictions) == 2


def test_save_cell_predictions(db, tile_state):
    eval_run = EvalRun.get_by_id(1)
    run_id = eval_run.id
    assert run_id == 1
    assert run_state_exists(1)

    # save nuclei and mark all tile states
    coords = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14]]
    save_pred_workings(run_id, coords)
    mark_finished_tiles(run_id, [0, 1, 2, 3])
    mark_nuclei_as_done(run_id)

    # set all nuclei to valid
    validate_pred_workings(run_id, coords)

    # move nuclei into predictions table
    commit_pred_workings(run_id)

    # save cell predictions
    cell_coords = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
    predictions = [1, 1, 2, 3, 1]
    save_cells(run_id, cell_coords, predictions)

    cell_predictions = Prediction.select()

    assert cell_predictions[0].x == 1
    assert cell_predictions[0].y == 2
    assert cell_predictions[0].cell_class == 1

    assert cell_predictions[1].x == 3
    assert cell_predictions[1].y == 4
    assert cell_predictions[1].cell_class == 1

    assert cell_predictions[2].x == 5
    assert cell_predictions[2].y == 6
    assert cell_predictions[2].cell_class == 2

    dicts = np.array(get_remaining_cells(run_id))

    assert dicts[0]['x'] == 11
    assert dicts[0]['y'] == 12
    assert dicts[1]['x'] == 13
    assert dicts[1]['y'] == 14
