from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
import numpy as np
import cv2
import sys
import os
import math
import matplotlib
import matplotlib.pyplot as plt
import random
from torch.utils.data.sampler import Sampler

from models.loss import dice_loss

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )   

def single_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )


def dilate(in_channels, out_channels, dilation):
    return nn.Conv2d(in_channels, out_channels, dilation)     

# UNet with dilations as implemented by Craig
# emil has added multi channel functionality (n_input) - 27-02-2022
class UNet(nn.Module):
    def __init__(self, n_class, n_input, channelsMultiplier):
        super().__init__()                
        self.dconv_down1 = double_conv(n_input, 44*channelsMultiplier)
        self.maxpool = nn.MaxPool2d(2)
        self.dconv_down2 = double_conv(44*channelsMultiplier, 44*2*channelsMultiplier)
        self.dconv_down3 = double_conv(44*2*channelsMultiplier, 44*4)       
        
        self.dilate1 = nn.Conv2d(44*4, 44*8, 3, dilation=1, padding=1)     
        self.dilate2 = nn.Conv2d(44*8, 44*8, 3, dilation=2, padding=2)     
        self.dilate3 = nn.Conv2d(44*8, 44*8, 3, dilation=4, padding=4)     
        self.dilate4 = nn.Conv2d(44*8, 44*8, 3, dilation=8, padding=8)     
        self.dilate5 = nn.Conv2d(44*8, 44*8, 3, dilation=16, padding=16)     
        self.dilate6 = nn.Conv2d(44*8, 44*8, 3, dilation=32, padding=32)     
        self.upsample = nn.Upsample(scale_factor=2)        

        self.sconv_up3 = single_conv(44*8, 44*4)
        self.sconv_up2 = single_conv(44*4, 44*2)
        self.sconv_up1 = single_conv(44*2, 44)

        self.dconv_up3 = double_conv(44*8, 44*4)
        self.dconv_up2 = double_conv(44*(2+2*channelsMultiplier), 44*2)
        self.dconv_up1 = double_conv(44*(2+(channelsMultiplier-1)), 44)        
        self.conv_last = nn.Conv2d(44, n_class, 1)
        
        
    def forward(self, x):        
        # to convert the Tensor to have the data type of floats
        x = x.float()
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        x1 = self.dilate1(x)
        x2 = self.dilate2(x1)
        x3 = self.dilate3(x2)
        x4 = self.dilate4(x3)
        x5 = self.dilate5(x4)
        x6 = self.dilate6(x5)
        # tried add_ does not work, also tried torch.add
        x = x1.add(x2).add(x3).add(x4).add(x5).add(x6)
        x = self.upsample(x)        
        x = self.sconv_up3(x)
        x = torch.cat([conv3, x], dim=1)        
        x = self.dconv_up3(x)
        x = self.upsample(x)   
        x = self.sconv_up2(x)     
        x = torch.cat([conv2, x], dim=1)       
        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = self.sconv_up1(x)     
        x = torch.cat([conv1, x], dim=1)           
        x = self.dconv_up1(x)        
        out = self.conv_last(x)        
        return out

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_unet(n_class, n_input, channelsMultiplier):
    model = UNet(n_class,n_input,channelsMultiplier)
    return model
