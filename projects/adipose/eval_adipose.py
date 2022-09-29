import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from happy.data.transforms.collaters import collater
from microscopefile import prediction_saver
from data.dataset.ms_dataset import SegDataset
from models.model import UNet
from happy.utils.graceful_killer import GracefulKiller
from happy.utils.utils import load_weights
from data.transforms.transforms import Normalizer, Resizer
from db.msfile_interface import get_msfile
import db.eval_runs_interface as db

# emil
import matplotlib.pyplot as plt

# emil first method that is being called in main file
# Load model weights and push to device
def setup_model(model_id, device, n_class, inputChannels, channelsMultiplier):
    # emil goes to the database and fetches the path of the weights associated with this model given by model_id
    # emil get_model_weights_by_id is a function in happy/db/models_training.py it uses the Model database that is defined in happy/db/models_training.py
    model_architecture, model_weights_path = db.get_model_weights_by_id(model_id)
    print(f"model pre_trained path: {model_weights_path}")
    if model_architecture.upper() == "UNET":
        # emil loads the model imported from happy/models/
        model = UNet(n_class, inputChannels, channelsMultiplier)
        numberDevice = device.split(":")[1]
        # emil
        # state_dict = torch.load(model_weights_path)
        model.load_state_dict(torch.load(model_weights_path))
        # Removes the module string from the keys if it's there
    else:
        raise ValueError(f"{model_architecture} not supported")
    # emil pushes model to specific GPU
    model = model.to(device)
    print("Pushed model to device")
    return model

# emil second method that is being called in main file
# Load dataset and dataloader
def setup_data(slide_id, run_id, model_id, batch_size, overlap, num_workers):
    # emil this return a Microsope object that has a Reader object inside it and data on the run (pixel_size, width, ...)
    # emil the Reader object is the one that actually opens the microscope file and returns the image
    # emil the Microscope object has a lot of methods for working with the microscope file
    # emil the data like seg_model_id is being fed to an EvalRun database (happy/db/eval_runs.py) that is created by the get_msfile function, where all the starting parameters are given 
    # emil hardcoded target pixelsize
    ms_file = get_msfile(
        slide_id=slide_id, run_id=run_id, seg_model_id=model_id, overlap=overlap, pixel_size = 0.2500,
    )
    # emil creates a PredictionSaver object from the happy/microscopefile/prediction_saver.py - it takes a microscope file as an input object 
    # emil it has several methods for storing the predictions to the databases - i need to change this to save the segmentation!!! and perhaps be able to do some post processing
    # emil also how predictions are saved in the database
    pred_saver = prediction_saver.PredictionSaver(ms_file)
    print("loading dataset")
    # emil
    remaining_data = np.array(db.get_remaining_tiles(ms_file.id))
    # emil creates a SegDataset object - this is essentially the Dataset object, that has the __iter__ method for the DataLoader to iterate over
    curr_data_set = SegDataset(
        # emil - remove Resizer as it is not needed for my analyses
        # ms_file, remaining_data, transform=transforms.Compose([Normalizer(), Resizer()])
        ms_file, remaining_data, transform=transforms.Compose([Normalizer()])
    )
    print("dataset loaded")
    print("creating dataloader")
    # emil loads it into a generic vanilla DataLoader
    dataloader = DataLoader(
        curr_data_set,
        num_workers=num_workers,
        collate_fn=collater,
        batch_size=batch_size,
    )
    print("dataloader ready")
    return dataloader, pred_saver


# Predict nuclei loop
def run_seg_eval(
    dataset, model, pred_saver, device, score_threshold=0.9,
):
    # object for graceful shutdown. Current loop finishes on SIGINT or SIGTERM
    killer = GracefulKiller()
    early_break = False
    tiles_to_evaluate = db.get_num_remaining_tiles(pred_saver.id)
    model.eval()
    with torch.no_grad():
        with tqdm(total=tiles_to_evaluate) as pbar:
            for batch in dataset:
                if not killer.kill_now:
                    # find the indices in the batch which are and aren't empty tiles
                    empty_mask = np.array(batch["empty_tile"])
                    tile_indexes = np.array(batch["tile_index"])
                    empty_inds = tile_indexes[empty_mask]
                    #non_empty_inds = tile_indexes[~empty_mask]
                    non_empty_inds = tile_indexes[~empty_mask]

                    # if there are empty tiles in the batch, save them as empty
                    if empty_inds.size > 0:
                        for empty_ind in empty_inds:
                            pbar.update()
                            pred_saver.save_empty([empty_ind])

                    # if there are non-empty tiles in the batch,
                    # eval model and save predictions
                    if non_empty_inds.size > 0:
                        # filter out indices without images
                        non_empty_imgs = np.array(
                            batch["img"].cpu().numpy()[~empty_mask]
                        )
                        # Get scale factor
                        scale = np.array(batch["scale"])[~empty_mask][0]

                        # emil - get if shifted tile
                        shifted_tiles = np.array(batch["shifted"])[~empty_mask]
                        print("")
                        print("")
                        print("shifted_tiles")
                        print(shifted_tiles)
                        print("")
                        print("")

                        # Network can't be fed batches of images
                        # as it returns predictions in one array
                        for i, non_empty_ind in enumerate(non_empty_inds):
                            # run network on non-empty images/tiles
                            input = torch.from_numpy(
                                np.expand_dims(non_empty_imgs[i], axis=0)
                            ).to(device)

                            print("np.shape(non_empty_imgs[i])")
                            print(np.shape(non_empty_imgs[i]))
                            # emil
                            label = torch.from_numpy(
                                np.expand_dims(np.zeros((1024,1024)), axis=0)
                            ).to(device)

                            # Predict
                            pred = model(input)
                            pred = torch.sigmoid(pred)
                            pred = pred.data.cpu().numpy()

                            # Correct predictions from resizing of img.
                            # boxes /= scale

                            # select indices which have a score above the threshold
                            pred_filtered = pred_saver.filter_by_score(
                                score_threshold, pred
                            )

                            # emil
                            print("np.shape(pred_filtered):")
                            print(np.shape(pred_filtered))

                            # emil saves predictions
                            plt.imsave("/well/lindgren/users/swf744/git/HAPPY/projects/adipose/tmpPred_"+str(tile_indexes[~empty_mask][i])+".png", pred_filtered[0])

                            pred_polygons = pred_saver.draw_polygons_from_mask(
                                # emil - this should probably be fixed
                                # pred_filtered, i
                                pred_filtered, i
                            )

                            pred_saver.save_seg(non_empty_ind, pred_polygons)
                            #pred_saver.save_seg(non_empty_ind, pred_polygons)
                            pbar.update()

                else:
                    early_break = True
                    break

    if not early_break and not pred_saver.file.segs_done:
        pred_saver.apply_nuclei_post_processing(cluster=True, remove_edges=True)
        pred_saver.commit_valid_nuclei_predictions()


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

