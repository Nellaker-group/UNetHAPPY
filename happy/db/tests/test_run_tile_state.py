import pathlib
import os

import pytest

from happy.db.base import init_db
from happy.db.eval_runs_interface import run_state_exists, save_new_tile_state, \
    get_run_state, mark_finished_tiles
from happy.db.tests.utils import setup_eval_run
from happy.db.eval_runs import TileState, EvalRun
from happy.db.base import database


# TODO: test if you can get the foreign key object by id or by the object

@pytest.fixture()
def db():
    db_name = f"pytest_temp_{__name__}.db"
    db_path = pathlib.Path(__file__).parent.absolute() / db_name
    init_db(db_path)
    yield
    os.remove(db_path)


@pytest.fixture()
def eval_run():
    setup_eval_run()
    yield


def test_temp_table(db, eval_run):
    assert not run_state_exists(1)

    eval_run = EvalRun.get_by_id(1)
    assert eval_run.id == 1

    tile_xy_list = [(0, 0), (0, 300), (0, 600), (300, 0)]
    save_new_tile_state(1, tile_xy_list)

    assert run_state_exists(1)

    tile_states = TileState.select(TileState).where(TileState.run == 1).dicts()

    assert [*tile_states[0]] == ['run', 'tile_index', 'tile_x', 'tile_y',
                                 'done']
    assert tile_states[0]['run'] == 1
    assert tile_states[0]['tile_index'] == 0
    assert tile_states[1]['tile_index'] == 1
    assert tile_states[1]['tile_y'] == 300
    assert tile_states[1]['done'] == False

    tile_xy_list = get_run_state(1)

    assert len(tile_xy_list) == 4
    assert len(tile_xy_list[0]) == 2
    assert tile_xy_list[0] == (0, 0)
    assert tile_xy_list[1] == (0, 300)


def test_save_new_tile_state(db, eval_run):
    tile_xy_list = [(0, 0), (0, 300), (0, 600), (300, 0)]
    save_new_tile_state(1, tile_xy_list)

    tile_states = TileState.select().where(TileState.run == 1)

    assert tile_states[0].done == False
    assert tile_states[2].done == False

    mark_finished_tiles(1, [0, 2])

    tile_states = TileState.select().where(TileState.run == 1)

    assert tile_states[0].done == True
    assert tile_states[1].done == False
    assert tile_states[2].done == True

    tile_state = TileState.select().where(
        (TileState.run == 1) & (TileState.tile_index == 1))

    assert tile_state[0].done == False

    mark_finished_tiles(1, [1])

    tile_state = TileState.select().where(
        (TileState.run == 1) & (TileState.tile_index == 1))

    assert tile_state[0].done == True

def test_get_all_completed_tiles(db, eval_run):
    tile_xy_list = [(0, 0), (0, 300), (0, 600), (300, 0)]
    save_new_tile_state(1, tile_xy_list)
    mark_finished_tiles(1, [0, 1, 2])

    with database.atomic():
        tile_coords = TileState.select().where(
            (TileState.run == 1) & (TileState.tile_index >= 0) &
            (TileState.tile_index < 3) & (TileState.done == True)).dicts()

    assert tile_coords[0]['done'] == True
    assert tile_coords[1]['done'] == True
    assert tile_coords[2]['done'] == True

    with database.atomic():
        tile_coords = TileState.select().where(
            (TileState.run == 1) & (TileState.tile_index >= 0) &
            (TileState.tile_index < 3) & (TileState.done == False)).dicts()

    assert not tile_coords
