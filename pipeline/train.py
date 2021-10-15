import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from IPython.display import clear_output
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from constants import *


def plot_history(train_history: list, val_history: list, title: str = 'loss') -> None:
    """Function for visualization of train process"""
    plt.figure()
    plt.title('{}'.format(title))
    plt.plot(train_history, label='train', zorder=1)

    points = np.array(val_history)
    steps = list(range(0, len(train_history) + 1, int(len(train_history) / len(val_history))))[1:]

    plt.scatter(steps, val_history, marker='+', s=180, c='orange', label='val', zorder=2)
    plt.xlabel('train steps')

    plt.legend(loc='best')
    plt.grid()

    plt.show()


def show_train_images():
    """Function shows images from train dataset"""
    df = pd.read_csv(TRAIN_DATAFRAME_PATH)
    cols = 8

    labels = df['label'].unique()
    rows = (len(labels) // cols) + 1

    fig, axs = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=(10, 50))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=1, wspace=None, hspace=None)

    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= len(labels):
                break

            row = df[df['label'] == labels[idx]].iloc[0]
            img = Image.open(os.path.join(FOLDER_PATH, row.filename.replace("/", "\ "[0])))
            axs[i, j].imshow(img)
            axs[i, j].set_facecolor('xkcd:salmon')
            axs[i, j].set_facecolor((1.0, 0.47, 0.42))
            axs[i, j].set_title(labels[idx])
            axs[i, j].get_xaxis().set_visible(False)
            axs[i, j].get_yaxis().set_visible(False)
            idx += 1


def test(
        model: nn.Module,
        val_dataloader: DataLoader,
        criterion: nn.Module,
        val_loss_log: list,
        device: str = DEVICE
) -> (float, float, float):
    """Function for test model"""
    model.eval()

    y_pred_val = np.array([])
    y_true_val = np.array([])
    val_loss = 0.
    val_pred = 0.
    val_size = 0

    with torch.no_grad():
        for i, (img, label) in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
            img, label = img.to(device), label.to(device).squeeze(1)
            pred = model(img)
            y_pred_val = np.concatenate([y_pred_val, pred.argmax(1).cpu().numpy()], axis=0)
            y_true_val = np.concatenate([y_true_val, label.cpu().numpy()], axis=0)
            loss = criterion(pred, label)
            val_loss += loss.item()
            val_size += pred.size(0)
            val_pred += (pred.argmax(1) == label).sum()

    val_loss /= val_size
    val_pred /= val_size
    f1_val = f1_score(y_true_val, y_pred_val, average='micro')
    val_loss_log.append(val_loss)
    return val_loss, val_pred, f1_val


def train_epoch(
        model: nn.Module,
        train_dataloader: DataLoader,
        criterion: nn.Module,
        opt: torch.optim.Optimizer,
        train_loss_log: list,
        device: str = DEVICE) -> (float, float, float):
    """Function for train one epoch"""
    model.train()

    y_pred = np.array([])
    y_true = np.array([])
    train_loss = 0.
    train_pred = 0.
    train_size = 0

    for i, (img, label) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        img, label = img.to(device), label.to(device).squeeze(1)
        pred = model(img)

        y_pred = np.concatenate([y_pred, pred.argmax(1).cpu().numpy()], axis=0)
        y_true = np.concatenate([y_true, label.cpu().numpy()], axis=0)

        loss = criterion(pred, label)

        opt.zero_grad()
        loss.backward()

        train_loss += loss.item()
        train_size += pred.size(0)
        train_loss_log.append(loss.data / pred.size(0))
        train_pred += (pred.argmax(1) == label).sum()

        opt.step()

    train_loss /= train_size
    train_pred /= train_size

    f1_train = f1_score(y_true, y_pred, average='micro')
    return train_loss, train_pred, f1_train


def train(
        model: nn.Module,
        criterion: nn.Module,
        opt: torch.optim.Optimizer,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        num_epoch: int = 15,
        device: str = DEVICE,
        save_checkpoint_path: str = 'point.ckpt',
        shed: torch.optim.lr_scheduler = None
) -> None:
    """Function for train model"""
    train_loss_log = []
    val_loss_log = []

    for epoch in range(num_epoch):
        print("Train is running |" + ('-' * epoch) + '>' + ('-' * (num_epoch - 1 - epoch)))
        print('* Epoch %d/%d' % (epoch + 1, num_epoch))
        ### TRAIN ###
        train_loss, train_acc, f1_train = train_epoch(model, train_dataloader, criterion, opt, train_loss_log, device)

        ### TEST ###
        val_loss, val_acc, f1_val = test(model, test_dataloader, criterion, val_loss_log, device)

        if shed:
            shed.step()

        clear_output()
        torch.save(model.state_dict(), save_checkpoint_path)
        print('f1_train', f1_train)
        print('f1_test', f1_val)
        plot_history(train_loss_log, val_loss_log, 'loss')
        print('Train loss:', train_loss * 100)
        print('Val loss:', val_loss * 100)
        print('Train acc:', train_acc * 100)
        print('Val acc:', val_acc * 100)
