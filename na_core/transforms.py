import importlib.util

import torch
from torchvision import transforms
import torchvision.transforms.functional as TF


PIL_AVAILABLE = importlib.util.find_spec("PIL") is not None
if PIL_AVAILABLE:
    from PIL import Image


PIL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ]
)

ATTACK_TFM = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
)


def preprocess(img):
    if PIL_AVAILABLE:
        return PIL_TRANSFORM(img)

    if not isinstance(img, torch.Tensor):
        raise TypeError("Unexpected image type without PIL")

    x = img.float().div(255.0)
    x = TF.resize(x, 256, antialias=True)
    x = TF.center_crop(x, [224, 224])

    mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype, device=x.device)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype, device=x.device)
    x = (x - mean[:, None, None]) / std[:, None, None]
    return x


def preprocess_x01(img):
    if PIL_AVAILABLE:
        return ATTACK_TFM(img)

    if not isinstance(img, torch.Tensor):
        raise TypeError("Unexpected image type without PIL")

    x = img.float().div(255.0)
    x = TF.resize(x, 256, antialias=True)
    x = TF.center_crop(x, [224, 224])
    return x
