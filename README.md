# Histology Analysis Pipeline.py (HAPPY)

For the training and evaluation of nuclei detection and cell classification
models on multi-organ histology whole slide images (WSI). Supported organs are:
placenta

In the future we hope to directly support liver, adipocyte and kidney histology, 
along with segmentation techniques.

Currently, a sqlite database (and Peewee orm) is used for the evaluation part of
the pipeline. The training part of the pipeline will be integrated with the
database soon.

## Project/Repo Structure

This repo is intended to supply all H&E multi-organ analysis. It is split up into 
'core' code which is (ideally) general enough to be used across organs and 'project'
code which is specific to a single project/organ. 

'project' code is found under `projects/{projectname}` with project names generally 
being named after the organ or study of interest (e.g. placenta or adipocyte). 
'core' code is everything else. Within 'core' there is also the source code, 
`happy`, which gets installed as a package for use anywhere. 

All project-specific code, entry points, datasets, results, analysis, etc, should live 
in the `projects/{projectname}` directory and use of any 'core' analysis should save the 
results into the project directory as well. Please see `projects/placenta` for an 
example.

Any changes made to 'core' code must be discussed and made via a branch and pull 
request into master. This pull request should contain only the specific changes to core
and will be subject to rigorous pull request review. Other project-specific 
changes which we may want to merge into master should also be made via a reviewed pull 
request but this review won't be so strict.

All checked in code is expected to be documented (public functions with Google style 
docstring), tested (pytest unit and integration tests), and formatted with Black code 
formatter. (Saying all of this I'm well aware that 'core' isn't fully compliant with 
this, and I'm working on it!)


## Setup

On first setup you will need to create and activate a conda environment. After this,
the make command will handle the rest of the setup (it may take a while).

### Rescomp:

This is for running training and evaluation on rescomp's GPUs and is the expected 
workflow since WSIs are stored on rescomp.

You will need to ssh into a GPU server before setup (this ensures GPU versions are 
installed correctly). If you haven't made a .condarc file, check the note bellow.

```bash
module load Anaconda3/2019.10
conda create -n {envname} python=3.7.2
conda activate {envname}
module load libvips/8.9.2-foss-2019a
export PATH=/{PathToYourUserDirWithSpace}/conda_stuff/my_envs/{envname}/bin/:$PATH
export JAVA_HOME=/usr/java/jdk1.8.0_112/
make environment_cu117
```

**Note:** if you are setting this up on rescomp with a teeny home directory, you will 
need to make a `.condarc` file at your home which tells conda where to put the installed
packages. Something like bellow:

```
envs_dirs:
    - /{PathToYourUserDirWithSpace}/conda_stuff/my_envs

pkgs_dirs:
    - /{PathToYourUserDirWithSpace}/conda_stuff/pkgs
```

### Local:

This is for running tests and bits of analysis locally. Assumes you're using a CPU.
If you want to use libvips locally you'll have to install the vips C binaries separately.
The installation methods depends on your OS, for more info follow the instructions 
here: https://github.com/libvips/libvips/wiki

```bash
conda create -n {envname} python=3.7.2
conda activate {envname}
make environment_cpu
```

I'm sure there will be some setup errors (state is a fickle beast) so let me know.

## How To Use

### Rescomp:

Before every use on rescomp you will need to load the modules, setup your path, and 
activate the conda environment (in that order). You can use a simple shell script for 
automating this:

```bash
source startup_script.sh
```

startup_script.sh:

```bash
#!/usr/bin/bash
module load libvips/8.9.2-foss-2019a
export PATH=/{PathToYourUserDirWithSpace}/conda_stuff/my_envs/{envname}/bin/:$PATH
eval "$(conda shell.bash hook)"
conda activate {envname}
```


### Cell Inference

For cell inference across a WSI use `python cell_inference.py`. 
This can be used to initialise a new run, continue an unfinished one, or do nuclei 
detection or cell classification separately. Each full inference run is treated as a 
separate entity and, as such, gets its own entry in the database. Using the unique 
run_id is the easiest way to access associated information about the run, including 
the model's predictions for that run.

Slides and their respective lab and patient information can be added to the
database with `happy/db/add_slides.py`. This needs to be done prior to
evaluation so that the code knows the slides' paths and metadata which will be used 
during the run.

Trained models will also need to be added to the database with `/happy/db/add_model.py`.

All model predictions are saved directly to the database and can be turned into
.tsv files that QuPath can read with `qupath/coord_to_tsv.py`.

Embedding vectors of the cell classifier, their class predictions, network confidence,
and WSI nuclei (x,y) coordinates are saved as an hdf5 file in 
`projects/{projectname}/results/embeddings/{lab_id}/{slide_name}/{run_id}.hdf5`.

### Cell Visualisation

To visualise the cell predictions without needing to load them into QuPath use
`projects/placenta/analysis/vis_cell_predictions.py`. This will create a plot of just
the predictions (with no histology tissue) at the highest resolution of the WSI. 

You can also visualise your training annotations ground truth nuclei locations using 
`analysis/visualisations/vis_ground_truth.py` or run your nuclei detector over 
images in your validation sets using `analysis/visualisations/vis_nuclei_preds.py`.

### Cell Embedding Analysis

To generate and visualise UMAP embeddings from WSI predictions of the cell
classifier look in `analysis/embeddings/`. Make sure to set the start index value and 
number of points to include in your embeddings. Indicies are saved sorted ascending by 
(x,y) so these values will take 'chunks' of predictions out of your image.

For most plotting needs use `/emebddings_interactive.py`. This will allow you to either 
save a png of the UMAP or create an html file which can be opened in a browser to 
interactively visualise the data. The plotting library swaps under the hood depending
on how many points you have so for many points use the non-interactive plot.

To combine two WSI into one embedding use `/joint_embeddings.py`, for just one cell 
class use `/single_class.py`, for filtering by network confidence predictions use 
`/single_confidence.py`, and for (small) 3d embeddings use `/embedding_3d.py`.

To visualise the top 100 outliers in the embedding vectors use 
`analysis/outliers/embeddings_outliers.py`. This can be useful for finding image 
artefacts or unusual cell types/features. Currently, this just prints the respective 
(x,y) coordinates and saves a figure containing the 100 outliers but more informative 
plotting might be added in the future.

### Cell Training

Main training scripts for nuclei detector and cell classifier are found in
`nuc_train.py` and `cell_train.py` respectively. As mentioned earlier, these have not 
currently been integrated with the database.

Raw training dataset images go in `/projects/{yourproject}/datasets/` and their 
groundtruth annotation csv files go in `/projects/{yourproject}/annotations/`. Separate 
training datasets can be combined into one training/validation dataset by the dataloader 
so long as the directory structure convention is followed.

Training metrics can be visualised using Visdom.

Training metrics, the specific parameters of the training run, and the
best model weights are saved to 
`/projects/{yourproject}/results/{model_type}/{exp_name}/{timestamp}/`.

### Making Training Data

After correcting model predictions in QuPath to generate training data, those 
corrections can be extracted to csv files using Groovy scripts in `/qupath`. 

You can then use `/happy/microscopefile/make_tile_dataset.py` to generate corresponding 
images and annotation csvs from the files generated by the Groovy scripts. These 
should be saved to your project directory.

### Tissues

This section of the readme will be filled out later! For now, you can look at the 
public repos happy and/or placenta to get an idea (or ask me).
