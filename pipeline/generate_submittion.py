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
    """Return empty """
    test_df = pd.read_csv(path).filename
    zero = pd.DataFrame({'filename': test_df, 'label': ['-'] * (test_df.shape[0])})
    zero = zero.set_index('filename')
    return zero


def generate_submit(
        model: nn.Module,
        model2: nn.Module,
        test_loader: DataLoader,
        name: str,
        device: str = DEVICE,
        visual: bool = False) -> np.ndarray:
    model.eval()
    model2.eval()

    counter = 0
    y_pred = np.array([])
    with torch.no_grad():
        for i, (filename, img) in enumerate(test_loader):
            img = img.to(device)

            pred = model(img)
            pred2 = model2(img)
            pred_all = pred + pred2
            arg_pred = pred_all.argmax(1).cpu().numpy()

            y_pred = np.concatenate([y_pred, arg_pred], axis=0)

    return y_pred
