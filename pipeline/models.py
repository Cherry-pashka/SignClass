from constants import *
from typing import Optional
import torch
from torchvision import models
import torch.nn as nn


def get_resnet_152(device: str = DEVICE,
                   ckpt_path: Optional[str] = None
                   ) -> nn.Module:
    model = models.resnet152(True)
    model.fc = nn.Sequential(nn.Linear(2048, 182))
    model = model.to(device)
    if ckpt_path:
        try:
            checkpoint = torch.load(ckpt_path)
            model.load_state_dict(checkpoint)
        except:
            print("Wrong checkpoint")
    return model


def get_densenet_121(device: str = DEVICE,
                     ckpt_path: Optional[str] = None
                     ) -> nn.Module:
    model = models.densenet121(True)
    model.classifier = nn.Sequential(nn.Linear(1024, 182))
    model = model.to(device)
    if ckpt_path:
        try:
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint)
        except:
            print("Wrong checkpoint")
    return model
