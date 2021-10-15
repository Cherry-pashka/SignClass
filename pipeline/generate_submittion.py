import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from constants import *


def generate_one_model_submit(
        model: nn.Module,
        test_loader: DataLoader,
        name: str,
        device: str = DEVICE) -> np.ndarray:
    """Returns labels predicted by one model"""
    y_pred = np.array([])
    filenames = []
    model.eval()
    with torch.no_grad():
        for i, (filename, img) in tqdm(enumerate(test_loader), total=len(test_loader)):
            img = img.to(device)
            pred = model(img)
            filenames += filename
            y_pred = np.concatenate([y_pred, pred.argmax(1).cpu().numpy()], axis=0)
            del pred, img
    return y_pred


def get_empty_submit(path: str) -> pd.DataFrame:
    """Returns empty submission"""
    test_df = pd.read_csv(path).filename
    zero = pd.DataFrame({'filename': test_df, 'label': ['-'] * (test_df.shape[0])})
    zero = zero.set_index('filename')
    return zero


def generate_submit(
        models: list,
        test_loader: DataLoader,
        name: str,
        device: str = DEVICE,
        visual: bool = False) -> np.ndarray:
    """Returns labels predicted by multiple models"""
    for i in range(len(models)):
        models[i].eval()

    y_pred = np.array([])
    with torch.no_grad():
        for i, (filename, img) in tqdm(enumerate(test_loader), total=len(test_loader)):
            img = img.to(device)
            pred_all = models[0](img)
            for model in models[1:]:
                pred_all += model(img)
            arg_pred = pred_all.argmax(1).cpu().numpy()
            y_pred = np.concatenate([y_pred, arg_pred], axis=0)

    return y_pred
