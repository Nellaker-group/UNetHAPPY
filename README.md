# Histology Analysis Pipeline.py (HAPPY)

For performing semantic segmentation of adipocytes in whole slide images (WSI).

Currently, a sqlite database (and Peewee orm) is used for the evaluation part of
the pipeline. The training part of the pipeline should be done using `https://github.com/Nellaker-group/PyTorchUnet`.
database at some point.

## Setup

On first setup you will need to create and activate a conda environment. After this,
the make command will handle the rest of the setup (it may take a while).

### Rescomp/BMRC:

This is for running evaluation on rescomp/BMRC's GPUs and is the expected 
workflow since WSIs are stored on rescomp/BMRC.

You will need to ssh into a GPU server before setup (this ensures GPU versions are 
installed correctly). If you haven't made a .condarc file, check the note bellow.

```bash
module load Anaconda3/2019.10
conda create -n {envname} python=3.10
conda activate {envname}
module load libvips/8.9.2-foss-2019a
export PATH=/{PathToYourUserDirWithSpace}/conda_stuff/my_envs/{envname}/bin/:$PATH
export JAVA_HOME=/usr/java/jdk1.8.0_112/
make environment_cu117
```

**Note:** if you are setting this up on rescomp/BMRC with a teeny home directory, you will 
need to make a `.condarc` file at your home which tells conda where to put the installed
packages. Something like:

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
conda create -n {envname} python=3.10
conda activate {envname}
make environment_cpu
```

I'm sure there will be some setup errors (state is a fickle beast) so let me know.

## How To Use

### Rescomp/BMRC:

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

### Training

Use `https://github.com/Nellaker-group/PyTorchUnet` for training - where there is also documentation on how to perform training.

### Making Training Data

See `https://github.com/Nellaker-group/PyTorchUnet` for how to make training data - where there is also documentation on how to perform training.

### Adipocyte Segmentation

For cell inference across a WSI use `python projects/adipose/eval_adipose_front.py`. 
This can be used to initialise a new segmentation run, continue an unfinished one.
Each full inference run is treated as a separate entity and, as such, gets its own entry in the database.
Using the unique run_id is the easiest way to access associated information about the run, including 
the model's predictions for that run.

Slides and their respective lab and patient information can be added to the
database with `projects/adipose/db/add_slides.py`. This needs to be done prior to
evaluation so that the code knows the slides' paths and metadata which will be used 
during the run. Trained models will also need to be added to the database with 
`projects/adipose/db/add_model.py`.

All model predictions are saved directly to the database and can be turned into
.geojson files with `projects/adipose/analysis/database_to_geojson.py`that store polygons of the segmentation
The .geojson files can be read into QuPath together with the WSI to visualise the segmentation.


