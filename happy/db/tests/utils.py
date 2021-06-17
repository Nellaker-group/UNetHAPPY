import random
import string

from happy.db.eval_runs_interface import save_new_tile_state
from happy.db.slides import Slide, Lab, Patient
from happy.db.models_training import TrainRun, Model
from happy.db.eval_runs import EvalRun


def random_string(n) -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=n))


def random_slide(patient, lab):
    slide = Slide.create(
        slide_name=random_string(10),
        tissue_type=random_string(10),
        lvl_x=random.randint(0, 10),
        pixel_size=random.random(),
        patient=patient,
        lab=lab,
    )
    return slide


def single_patient():
    patient = Patient.create(diagnosis="blah", clinical_history="blah blah")
    return patient


def single_lab():
    lab = Lab.create(
        country="test_place", primary_contact="Test Contact", slides_dir="test/dir"
    )
    return lab


def random_testing_lab():
    lab = Lab.create(
        country=random_string(5),
        primary_contact=random_string(7),
        slides_dir=random_string(10),
    )
    return lab


def single_train_run():
    train_run = TrainRun.create(
        run_name="test",
        type="nuclei",
        pre_trained_path="coco",
        num_epochs=5,
        batch_size=5,
        init_lr=0.0001,
        lr_step=8,
    )
    return train_run


def single_nuc_model(train_run):
    nuc_model = Model.create(
        train_run=train_run,
        type=train_run.type,
        path="path/to/model",
        architecture="resnet-50",
        performance=0.81,
    )
    return nuc_model


def setup_eval_run():
    lab = single_lab()
    patient = single_patient()
    slide = random_slide(patient, lab)
    train_run = single_train_run()
    nuc_model = single_nuc_model(train_run)
    eval_run = EvalRun.create(nuc_model=nuc_model, slide=slide)
    return eval_run


def setup_tile_state():
    tile_xy_list = [(0, 0), (0, 300), (0, 600), (300, 0)]
    save_new_tile_state(1, tile_xy_list)
