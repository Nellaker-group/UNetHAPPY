import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
import numpy as np
from torch.utils.data.sampler import Sampler
import os

from Dataset import GetDataMontage, GetDataFolder
from Sampler import MontageSamplerUniform, MontageSamplerUniformFrankenstein, MontageSamplerDatasetSize, ImageSamplerUniform, ImageSamplerUniformFrankenstein, ImageSamplerUniformFrankensteinV2, ImageSamplerUniformFrankensteinEmil, ImageSamplerDatasetSize, ValSampler, ImageValSampler

def get_dataloader(trainDir, valDir,imageDir,preName,ifAugment,noTiles,augSeed,ifSizeBased,frank,inputChannels,normFile,input512,zoomFile):

    # use the same transformations for train/val in this example
    trans = transforms.Compose([
        transforms.ToTensor(),
    ])

    print("imageDir:")
    print(imageDir)

    if imageDir:
        # read in data
        train_set = GetDataFolder("train", preName, augSeed, frank, inputChannels, normFile, input512, zoomFile, pathDir=trainDir, transform=trans, ifAugment=ifAugment)
        val_set = GetDataFolder("validation", preName, augSeed, frank, inputChannels, normFile, input512, zoomFile, pathDir=valDir, transform=trans)
    else:
        # read in data
        train_set = GetDataMontage("train", preName, augSeed, frank, normFile, input512, pathDir=trainDir, transform=trans, ifAugment=ifAugment)
        val_set = GetDataFolder("validation", preName, augSeed, frank, inputChannels, normFile, input512, pathDir=valDir, transform=trans)

    image_datasets = {
        'train': train_set, 'val': val_set
    }

    sample_size_train = noTiles
    # it uses all val tiles
    batch_size = 2
    input_size = 1024
    dataloaders = {}

    if input512 == 1:
        batch_size = 8
        input_size = 512
    
    if imageDir:
        # read in data
        if ifSizeBased == 1:
            samplie_train = ImageSamplerDatasetSize(train_set, sample_size_train, input_size, 0)
        elif frank == 1:
            samplie_train = ImageSamplerUniformFrankensteinV2(train_set, sample_size_train, input_size, 0)
        elif frank == 2:
            samplie_train = ImageSamplerUniformFrankensteinEmil(train_set, sample_size_train, input_size, 0)
        else:
            samplie_train = ImageSamplerUniform(train_set, sample_size_train, input_size, 0)
        samplie_val = ImageValSampler(val_set,  input_size, 0)
        dataloaders = {
            'train': DataLoader(train_set, batch_size=batch_size, num_workers=0, sampler=samplie_train),
            'val': DataLoader(val_set, batch_size=batch_size, num_workers=0, sampler=samplie_val)
        }
    else:
        # read in data
        if ifSizeBased==1:
            samplie_train = MontageSamplerDatasetSize(train_set, sample_size_train, input_size, 0)
        elif frank == 1:
            samplie_train = MontageSamplerUniformFrankenstein(train_set, sample_size_train, input_size, 0)
        else:
            samplie_train = MontageSamplerUniform(train_set, sample_size_train, input_size, 0)
        samplie_val = ImageValSampler(val_set, input_size, 0)
        dataloaders = {
            'train': DataLoader(train_set, batch_size=batch_size, num_workers=0, sampler=samplie_train),
            'val': DataLoader(val_set, batch_size=batch_size, num_workers=0, sampler=samplie_val)
        }

    return dataloaders
