from collections import OrderedDict

import pretrainedmodels
import torch
import torch.nn as nn


def build_inceptionresnetv2(out_features=5):
    model = pretrainedmodels.__dict__["inceptionresnetv2"](
        num_classes=1001, pretrained="imagenet+background"
    )

    layers = OrderedDict()
    layers["linear_layer_1"] = torch.nn.Linear(
        in_features=1536, out_features=768, bias=True
    )
    layers["linear_Re_lu_1"] = torch.nn.ReLU()
    layers["linear_dropout_1"] = torch.nn.Dropout(p=0.3)
    layers["linear_layer_2"] = torch.nn.Linear(
        in_features=768, out_features=384, bias=True
    )
    layers["linear_Re_lu_2"] = torch.nn.ReLU()
    layers["linear_dropout_2"] = torch.nn.Dropout(p=0.3)
    layers["embeddings_layer"] = torch.nn.Linear(
        in_features=384, out_features=64, bias=True
    )
    layers["output_layer"] = torch.nn.Linear(in_features=64, out_features=out_features)

    new_module = nn.Sequential(layers)
    model.last_linear = new_module

    for child in model.children():
        for param in child.parameters():
            param.requires_grad = True

    return model
