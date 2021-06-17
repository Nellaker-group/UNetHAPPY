import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import h5py

from happy.db.msfile_interface import get_msfile
from happy.microscopefile.prediction_saver import PredictionSaver
from happy.data.dataset.ms_dataset import CellDataset
from happy.data.transforms.collaters import cell_collater
from happy.utils.graceful_killer import GracefulKiller
from happy.data.transforms.transforms import Normalizer, Resizer
from happy.models.model_builder import build_cell_classifer
import happy.db.eval_runs_interface as db


# Load model weights and push to cuda device
def setup_model(model_id, out_features):
    torch_home = Path(__file__).parent.parent.parent.absolute()
    os.environ["TORCH_HOME"] = str(torch_home)

    model_architecture, model_weights_path = db.get_model_weights_by_id(model_id)
    print(f"model pre_trained path: {model_weights_path}")

    model = build_cell_classifer(model_architecture, out_features)
    model.load_state_dict(torch.load(model_weights_path), strict=True)

    model = torch.nn.DataParallel(model).cuda()
    print("Pushed model to cuda")
    return model, model_architecture


# Load dataset and dataloader
def setup_data(
    run_id, model_id, model_architecture, batch_size, num_workers, cell_saving=True
):
    ms_file = get_msfile(run_id=run_id, cell_model_id=model_id)
    pred_saver = PredictionSaver(ms_file)
    print("loading dataset")
    image_size = (224, 224) if model_architecture == "resnet-50" else (299, 299)
    if not cell_saving:
        remaining_data = np.array(db.get_all_prediction_coordinates(run_id))
    else:
        remaining_data = np.array(db.get_remaining_cells(run_id))
    dataset = CellDataset(
        ms_file,
        remaining_data,
        transform=transforms.Compose(
            [
                Normalizer(),
                Resizer(
                    min_side=image_size[0],
                    max_side=image_size[1],
                    padding=False,
                    scale_annotations=False,
                ),
            ]
        ),
    )
    print("dataset loaded")
    print("creating dataloader")
    dataloader = DataLoader(
        dataset,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=cell_collater,
        batch_size=batch_size,
    )
    print("dataloader ready")
    return dataloader, pred_saver


# Setup or get path to embeddings hdf5 save location
def setup_embedding_saving(project_name, run_id, cell_saving=True):
    embeddings_dir = Path(__file__).parent.parent.parent / "Results" / "embeddings"
    path = db.get_embeddings_path(run_id, embeddings_dir)
    embeddings_path =  embeddings_dir / path
    if not os.path.isfile(embeddings_path):
        total_cells = db.get_total_num_nuclei(run_id)
        with h5py.File(embeddings_path, "w-") as f:
            f.create_dataset("predictions", (total_cells,), dtype="int8")
            f.create_dataset("embeddings", (total_cells, 64), dtype="float32")
            f.create_dataset("confidence", (total_cells,), dtype="float16")
            f.create_dataset("coords", (total_cells, 2), dtype="uint32")
    elif not cell_saving:
        raise ValueError(
            "Embeddings file already exists. Please move it for a no save run"
        )
    return embeddings_path


# Predict cell classes loop
def run_cell_eval(
    dataset, cell_model, pred_saver, embeddings_path, cell_saving=True
):
    # object for graceful shutdown. Current loop finishes on SIGINT or SIGTERM
    killer = GracefulKiller()
    early_break = False
    if not cell_saving:
        remaining = len(db.get_all_prediction_coordinates(pred_saver.id))
    else:
        remaining = db.get_num_remaining_cells(pred_saver.id)
    cell_model.eval()

    def copy_data(module, input, output):
        embedding.copy_(output.data)

    with torch.no_grad():
        with tqdm(total=remaining) as pbar:
            for i, batch in enumerate(dataset):
                if not killer.kill_now:
                    # evaluate model and set up saving the embeddings layer
                    embedding = torch.zeros((batch["img"].shape[0], 64))
                    handle = (
                        cell_model.module.fc.embeddings_layer.register_forward_hook(
                            copy_data
                        )
                    )
                    # Calls forward() and copies the embedding data
                    class_prediction = cell_model(batch["img"].cuda().float())
                    # Removes the hook before the next forward() call
                    handle.remove()

                    # get predictions, confidence, and embeddings
                    _, predicted = torch.max(class_prediction.data, 1)
                    confidence = softmax(class_prediction.data, dim=1).cpu().numpy()
                    predicted = predicted.cpu().tolist()
                    embeddings = embedding.cpu().tolist()
                    top_confidence = confidence[(range(len(predicted)), [predicted])][0]

                    # setup values for saving in hdf5 file
                    num_to_save = len(predicted)
                    start = -remaining
                    end = -remaining + num_to_save

                    # save embeddings layer for each prediction in the batch
                    with h5py.File(embeddings_path, "r+") as f:
                        if end == 0:
                            end = len(f["predictions"])
                        f["predictions"][start:end] = predicted
                        f["embeddings"][start:end] = embeddings
                        f["confidence"][start:end] = top_confidence
                        f["coords"][start:end] = batch["coord"]
                    remaining -= num_to_save

                    if cell_saving:
                        # save the class predictions of the batch
                        pred_saver.save_cells(batch["coord"], predicted)
                    pbar.update(dataset.batch_size)
                else:
                    early_break = True
                    break

    if not early_break and cell_saving:
        pred_saver.finished_cells()


def clean_up(pred_saver):
    ms_file = pred_saver.file
    try:
        if isinstance(ms_file.reader, ms_file.reader.BioFormatsFile):
            print("shutting down BioFormats vm")
            ms_file.reader.stop_vm()
        else:
            pass
    except AttributeError:
        pass
