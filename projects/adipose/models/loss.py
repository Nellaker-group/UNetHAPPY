from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    
    intersection = (pred * target).sum(dim=2).sum(dim=2)    
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    return loss.mean()

def calc_loss(pred, target, metrics, bce_weight=0.5):
    
    # bce and bce1 are the same
    # bce1 = F.binary_cross_entropy_with_logits(pred, target)
    pred = torch.sigmoid(pred)
    bce = F.binary_cross_entropy(pred, target)
    
    dice = dice_loss(pred, target)
    loss = bce * bce_weight + dice * (1 - bce_weight)
    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    return loss


# like glasonbury does it
def calc_lossCraig(pred, target, metrics):
    
    # bce and bce1 are the same
    # bce1 = F.binary_cross_entropy_with_logits(pred, target)
    pred = torch.sigmoid(pred)
    bce = F.binary_cross_entropy(pred, target)
    
    dice = dice_loss(pred, target)
    loss = bce + dice 
    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    return loss
