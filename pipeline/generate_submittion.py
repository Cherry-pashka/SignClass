import pandas as pd
from constants import *
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import numpy as np
from PIL import Image
from datasets import get_label_replacers


def get_empty_submit(path: str) -> pd.DataFrame:
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
        visual: bool = False) -> None:
    model.eval()
    model2.eval()

    counter = 0
    y_pred = np.array([])
    with torch.no_grad():
        for i, (filename, img) in enumerate(test_loader):
            img = img.to(device)
            pred = model(img)
            pred2 = model2(img)
            res2 = pred2.sort(descending=True).values[:, 0] - pred2.sort(descending=True).values[:, 1]
            inds2 = pred2.sort(descending=True).indices[:, 0:5]

            res = pred.sort(descending=True).values[:, 0] - pred.sort(descending=True).values[:, 1]
            inds = pred.sort(descending=True).indices[:, 0:5]
            for j in range(len(res)):
                if pred.argmax(1).cpu().numpy()[j] != pred2.argmax(1).cpu().numpy()[j] or res[j] < 2 or res2[j] < 2:
                    counter += 1
                    if visual:
                        print(filename[j].split("/")[1], '           ', res[j], '  ', inds[j])
                        print(filename[j].split("/")[1], '           ', res2[j], '  ', inds2[j])
                        display(Image.open(f'{FOLDER_PATH}/{filename[j]}'))

            y_pred = np.concatenate([y_pred, pred.argmax(1).cpu().numpy()], axis=0)
    zero = get_empty_submit(SAMPLE_SUBMISSION_PATH)
    zero.label = y_pred
    label2int, int2label = get_label_replacers(TRAIN_DATAFRAME_PATH)
    zero.label = zero.label.replace(int2label)
    zero.to_csv(f'{name}_submit.csv')
    print(counter)
