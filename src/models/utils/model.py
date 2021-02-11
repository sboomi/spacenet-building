import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger('utils_model')


def load_latest_model(model_folder: Path) -> Optional[Path]:
    """Finds the latest model inside the folder where
    all the models are saved

    Args:
        model_folder (Path): folder containing the models

    Returns:
        Path: file path of the most recent model
    """

    model_cps = [
        filename
        for filename in model_folder.parent.iterdir()
        if filename.name.startswith(model_folder.name)
    ]

    if len(model_cps) == 0:
        return None

    return sorted(model_cps)[-1]


def format_batch_score(batch: int, loss: float) -> str:
    """Formats the current result withitn batch processing
    in a string used for logging.

    Args:
        batch (int): current batch
        loss (float): current associated loss

    Returns:
        str: formatted string
    """
    return f"Batch {batch}: {loss}"


def format_epoch_score(epoch: int, loss: float) -> str:
    """Formats the results obtained at the end of an epoch in
    a string used for logging.

    Args:
        epoch (int): current epoch
        loss (float): current associated loss

    Returns:
        str: formatted string
    """
    return f"Epoch {epoch}: {loss}"


# TODO: needs typing and docstring
@torch.no_grad()
def evaluate(model, loader, device, criterion, n_labels, n_batch=-1):
    model.eval()

    val_loss = 0
    cm = np.zeros((n_labels, n_labels))

    total = len(loader) if n_batch == -1 else n_batch
    for bs, ((img, lab), _) in tqdm(
        enumerate(loader),
        desc='Eval',
        total=total
    ):
        if bs == n_batch:
            break

        img, lab = img.to(device), lab.to(device)

        # _, clf_out = model(img)
        clf_out = model(img)
        loss = criterion(clf_out, lab)

        val_loss += loss.detach().item()

        cm += confusion_matrix(clf_out.argmax(dim=1).flatten().cpu().numpy(),
                               lab.flatten().cpu().numpy(),
                               labels=list(range(n_labels)))

    return val_loss / total, cm


def train_per_epoch(model:        nn.Module,
                    train_loader: DataLoader,
                    criterion:    nn.modules.loss,
                    optimizer:    Optimizer,
                    device:       torch.device
                    ) -> None:
    """Trains a model for a single epoch.

    Args:
        model: the model to train
        train_loader: data loader of the train dataset
        criterion: the loss function
        optimizer: method for optimization of the model parameters
        device: the device on which to store the tensors
    """
    model.train()

    for bs, ((img, lab), _) in tqdm(
        enumerate(train_loader),
        desc='Batch',
        total=len(train_loader)
    ):
        img, lab = img.to(device), lab.to(device)

        optimizer.zero_grad()

        # _, clf_out = model(img)
        clf_out = model(img)
        loss = criterion(clf_out, lab)

        loss.backward()
        optimizer.step()

        if (bs+1) % 100 == 0:
            logger.info(format_batch_score(bs + 1, loss.detach().item()))


# TODO: improve docstring
def train(model:        nn.Module,
          train_loader: DataLoader,
          val_loader:   DataLoader,
          criterion:    nn.modules.loss,
          optimizer:    Optimizer,
          epochs:       int,
          device:       torch.device,
          n_labels:     int,
          model_folder: Path
          ) -> dict[str, dict[str, list]]:
    """Main function for training."""

    results = {"train": {"loss": [], "conf_matrix": [], "accuracy": []},
               "val":   {"loss": [], "conf_matrix": [], "accuracy": []}}
    model.train()
    chkpt_folder = model_folder / 'checkpoints'
    chkpt_folder.mkdir(exist_ok=True)

    for epoch in tqdm(range(epochs), desc='Epoch'):
        train_per_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        train_loss, train_cm = evaluate(
            model,
            train_loader,
            device,
            criterion,
            n_labels,
            len(val_loader)
        )
        val_loss, val_cm = evaluate(
            model,
            val_loader,
            device,
            criterion,
            n_labels
        )

        train_acc = (train_cm * np.identity(train_cm.shape[0])).sum() / train_cm.sum()
        val_acc = (val_cm * np.identity(val_cm.shape[0])).sum() / val_cm.sum()

        # Log the metrics
        results["train"]["loss"].append(train_loss)
        results["train"]["conf_matrix"].append(train_cm)
        results["train"]["accuracy"].append(train_acc)
        results["val"]["loss"].append(val_loss)
        results["val"]["conf_matrix"].append(val_cm)
        results["val"]["accuracy"].append(val_acc)

        logger.info(
            f'[{epoch:3d}] Train loss: {train_loss:5.3f}, '
            f'Train accuracy: {100 * train_acc:5.3f}%'
        )
        logger.info(
            f'[{epoch:3d}] Val loss: {val_loss:5.3f}, '
            f'Val accuracy: {100 * val_acc:5.3f}%'
        )

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'results': results},
            chkpt_folder / f'tiramisu_chkpt_epoch_{epoch:03d}.pt'
        )

    return results

