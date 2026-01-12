from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models


def build_model():
    try:
        model = models.vgg16(weights=None)
    except TypeError:
        model = models.vgg16(pretrained=False)

    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(4096, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(512, 10),
    )
    return model


def load_state_dict(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")
