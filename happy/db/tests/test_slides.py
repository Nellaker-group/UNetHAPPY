import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from happy.db.base import init_db
from happy.db.slides import Patient
from happy.db.tests.utils import random_slide, random_testing_lab, single_lab


@pytest.fixture()
def db():
    temp_dir = TemporaryDirectory()
    db_name = f"pytest_temp_{__name__}.db"
    db_path = Path(temp_dir.name) / db_name
    init_db(db_path)
    yield
    os.remove(db_path)


@pytest.fixture()
def patient_with_slide():
    patient = Patient.create(diagnosis="foo", clinical_history="bar")
    lab = single_lab()
    random_lab = random_testing_lab()
    random_slide(patient, lab)
    random_slide(patient, lab)
    random_slide(patient, random_lab)
    yield


def test_patient_slides(db, patient_with_slide):
    patient = Patient.get(Patient.id == 1)

    assert len(patient.slides) == 3


def test_add_patient(db):
    # Save a patient
    patient = Patient.create(diagnosis="bad", clinical_history="also bad")
    assert patient.id == 1

    # Get the same patient from db and check they're the same
    old_patient = Patient.get(Patient.id == 1)
    assert patient == old_patient

    # Change the patient diagnosis and update db
    old_patient.diagnosis = "good"
    old_patient.save()

    # Get the updated patient back and check it's different to the old object
    saved_patient = Patient.get(Patient.id == 1)
    assert patient.diagnosis != saved_patient.diagnosis
    assert old_patient.diagnosis == saved_patient.diagnosis
    assert saved_patient.diagnosis == "good"

    # Make a second patient and add them to the db
    patient2 = Patient(diagnosis="super bad", clinical_history="even worse")
    patient2.save()

    # Get this new patient and compare them to previous patient
    new_patient = Patient.get(Patient.diagnosis == "super bad")
    assert new_patient != old_patient
    assert new_patient.diagnosis == "super bad"
