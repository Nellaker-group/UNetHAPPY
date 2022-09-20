#!/bin/bash

## creates the database here:
## /gpfs3/well/lindgren/users/swf744/git/HAPPY/happy/db/main.db

python db/add_model.py --path-to-model /gpfs3/well/lindgren/users/swf744/git/PyTorchUnet/weights/weightsSeqTiles_ep200t2022_07_28-174047g0.5s865au1op1st30sB0LR0.0001fr1ch3si1024zo1mu1.dat --model-performance 0.85 --run-name test --run-type test --path-to-pretrained-model /gpfs3/well/lindgren/users/swf744/git/PyTorchUnet/weights/weightsSeqTiles_ep200t2022_07_28-174047g0.5s865au1op1st30sB0LR0.0001fr1ch3si1024zo1mu1.dat --num-epochs 200 --batch-size 2 --init-lr 0.0001 --lr-step 30 --model-architecture UNet

python db/add_slides.py --slides-dir /gpfs3/well/lindgren/users/swf744/adipocyte/data/WSIexamples/ENDOXexamples/ --lab-country USA --primary-contact Person --slide-file-format .scn --pixel-size 0.2500
