import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from happy.db.base import init_db
from happy.db.tests.utils import setup_eval_run
from happy.db.eval_runs import EvalRun
from happy.db.eval_runs_interface import get_embeddings_path


def _setup_temp_dir():
    return TemporaryDirectory()


@pytest.fixture()
def db():
    temp_dir = _setup_temp_dir()
    db_name = f"pytest_temp_{__name__}.db"
    db_path = Path(temp_dir.name) / db_name
    init_db(db_path)
    yield
    os.remove(db_path)


@pytest.fixture()
def eval_run():
    setup_eval_run()
    yield



def test_get_embeddings_path(db, eval_run):
    temp_dir = _setup_temp_dir()
    eval_run = EvalRun.get_by_id(1)
    expected_embeddings_path = f"lab_1/slide_{eval_run.slide.slide_name}/run_1.hdf5"

    embeddings_path = get_embeddings_path(1, f"{temp_dir.name}results/embeddings")

    assert expected_embeddings_path == embeddings_path

