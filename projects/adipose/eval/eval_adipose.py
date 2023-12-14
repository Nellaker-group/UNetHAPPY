import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from happy.data.transforms.collaters import collater
from happy.utils.utils import GracefulKiller
from projects.adipose.microscopefile import prediction_saver
from projects.adipose.data.transforms.transforms import Normalizer
from projects.adipose.db.msfile_interface import get_msfile
import projects.adipose.db.eval_runs_interface as db
from projects.adipose.data.dataset.ms_dataset import SegDataset
from projects.adipose.models.model import UNet


# Load model weights and push to device
def setup_model(model_id, device, n_class, inputChannels, channelsMultiplier):
    # goes to the database and fetches the path of the weights associated with this model given by model_id using get_model_weights_by_id() from happy/db/models_training.py 
    model_architecture, model_weights_path = db.get_model_weights_by_id(model_id)
    print(f"model pre_trained path: {model_weights_path}")
    if model_architecture.upper() == "UNET":
        # loads the model imported from happy/models/
        model = UNet(n_class, inputChannels, channelsMultiplier)
        model.load_state_dict(torch.load(model_weights_path,map_location=device))
    else:
        raise ValueError(f"{model_architecture} not supported")
    #  pushes model to specific GPU
    model = model.to(device)
    print("Pushed model to device")
    return model



# Load dataset and dataloader
def setup_data(slide_id, run_id, model_id, batch_size, overlap, num_workers, pixel_size):
    # returns a Microsope object (methods for workign with microscope file) that has a Reader object (for opening the slide) inside it and data on the run (pixel_size, width, ...)
    ms_file = get_msfile(
        slide_id=slide_id, run_id=run_id, seg_model_id=model_id, overlap=overlap, pixel_size = pixel_size
    )
    # creates a PredictionSaver object (metohds for storing the predictions to the databases) from the happy/microscopefile/prediction_saver.py - it takes a microscope file as an input object 
    pred_saver = prediction_saver.PredictionSaver(ms_file)
    print("loading dataset")
    remaining_data = np.array(db.get_remaining_tiles(ms_file.id))
    # creates a SegDataset object - this is essentially the Dataset object, that has the __iter__ method for the DataLoader to iterate over
    curr_data_set = SegDataset(
        # removed Resizer as it is not needed for my analyses
        ms_file, remaining_data, transform=transforms.Compose([Normalizer()])
    )
    print("dataset loaded")
    print("creating dataloader")
    # loads it into a generic vanilla DataLoader
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
    dataset, model, pred_saver, device, write_geojson, score_threshold=0.9, 
):
    # object for graceful shutdown. Current loop finishes on SIGINT or SIGTERM
    killer = GracefulKiller()
    early_break = False
    tiles_to_evaluate = db.get_num_remaining_tiles(pred_saver.id)
    poly_id = 0
    model.eval()
    with torch.no_grad():
        for batch in dataset:
    
            if not killer.kill_now:
                # find the indices in the batch which are and aren't empty tiles
                empty_mask = np.array(batch["empty_tile"])
                tile_indexes = np.array(batch["tile_index"])
                empty_inds = tile_indexes[empty_mask]
                non_empty_inds = tile_indexes[~empty_mask]

                # if there are empty tiles in the batch, save them as empty
                if empty_inds.size > 0:
                    for empty_ind in empty_inds:
                        pred_saver.save_empty([empty_ind])

                # if there are non-empty tiles in the batch,
                # eval model and save predictions
                if non_empty_inds.size > 0:
                    # filter out indices without images
                    non_empty_imgs = np.array(
                        batch["img"].cpu().numpy()[~empty_mask]
                    )

                    # Network can't be fed batches of images
                    # as it returns predictions in one array
                    for i, non_empty_ind in enumerate(non_empty_inds):
                        
                        # run network on non-empty images/tiles
                        input = torch.from_numpy(
                            np.expand_dims(non_empty_imgs[i], axis=0)
                        ).to(device)

                        # Predict
                        pred = model(input)
                        pred = torch.sigmoid(pred)
                        pred = pred.data.cpu().numpy()

                        # creates binary mask of entries with a score above the threshold
                        pred_filtered = pred_saver.filter_by_score(
                            score_threshold, pred
                        )

                        # creates shapely polygons from the mask using find_contours() from skimage
                        pred_polygons = pred_saver.draw_polygons_from_mask(
                            pred_filtered, i
                        )

                        # saves the polygon to the UnvalidatedPrediction table 
                        pred_saver.save_seg(non_empty_ind, pred_polygons, poly_id)                        
                        # for keeping track of the poly_id each polygon across the slide has a unique poly_id
                        poly_id += len(pred_polygons)

            else:
                early_break = True
                break

    if not early_break and not pred_saver.file.seg_done:
        # merges the polygons and saves them to the Prediction table
        pred_saver.apply_seg_post_processing(write_geojson, overlap=True)

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

