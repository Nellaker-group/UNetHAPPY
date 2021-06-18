from pathlib import Path
from tempfile import TemporaryDirectory
import os
import time

import pytest
from peewee import chunked, Tuple, ValuesList, EnclosedNodeList

from happy.db.base import init_db, database
from happy.db.eval_runs import Prediction, UnvalidatedPrediction
from happy.db.tests.utils import setup_eval_run


@pytest.fixture()
def db():
    temp_dir =  TemporaryDirectory()
    db_name = f"pytest_temp_{__name__}.db"
    db_path =  Path(temp_dir.name) / db_name
    init_db(db_path)
    yield
    os.remove(db_path)


@pytest.fixture()
def eval_run():
    setup_eval_run()
    yield


@pytest.mark.integration
def test_predictions(db, eval_run):
    total_time = 0
    n = 1500000
    batch_size = 10000

    data = []
    for i in range(n):
        data.append({"run": 1, "x": i, "y": i})

    with database.atomic():
        for batch in chunked(data, batch_size):
            start = time.time()
            UnvalidatedPrediction.insert_many(batch).execute()
            end = time.time()
            duration = end - start
            total_time += duration

    print("")
    print("Insert many:")
    print(f"average time taken per batch: {total_time / (n / batch_size):.4f} ms")
    print(f"total time: {total_time:.4f} s")

    # ---------

    coords = [[d["x"], d["y"]] for d in data]
    coords.pop()
    batch = 100000

    start = time.time()
    with database.atomic():
        for i in range(0, len(coords), batch):
            coords_vl = ValuesList(coords[i : i + batch], columns=("x", "y"))
            rhs = EnclosedNodeList([coords_vl])
            UnvalidatedPrediction.select().where(
                (UnvalidatedPrediction.run == 1)
                & (Tuple(UnvalidatedPrediction.x, UnvalidatedPrediction.y) << rhs)
            )

    end = time.time()
    duration = end - start

    print("")
    print("Select all:")
    print(f"total time: {duration:.4f} s")

    # ---------

    batch = 100000
    start = time.time()
    with database.atomic():
        for i in range(0, len(coords), batch):
            coords_vl = ValuesList(coords[i : i + batch], columns=("x", "y"))
            rhs = EnclosedNodeList([coords_vl])
            query = UnvalidatedPrediction.update(is_valid=True).where(
                (UnvalidatedPrediction.run == 1)
                & (Tuple(UnvalidatedPrediction.x, UnvalidatedPrediction.y) << rhs)
            )
            query.execute()

        end = time.time()
        duration = end - start

    print("")
    print("Update all:")
    print(f"total time: {duration:.4f} s")

    # ---------

    start = time.time()
    source = (
        UnvalidatedPrediction.select(
            UnvalidatedPrediction.run, UnvalidatedPrediction.x, UnvalidatedPrediction.y
        )
        .where(
            (UnvalidatedPrediction.run == 1) & (UnvalidatedPrediction.is_valid == True)
        )
        .order_by(UnvalidatedPrediction.x, UnvalidatedPrediction.y.asc())
    )

    rows = Prediction.insert_from(
        source, fields=[Prediction.run, Prediction.x, Prediction.y]
    ).execute()

    end = time.time()
    duration = end - start

    print("")
    print("Insert into:")
    print(f"total time to insert {rows} rows: {duration:.4f} s")
