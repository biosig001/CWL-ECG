import math
import random

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from config import (
    BATCH_SIZE,
    LEARNING_RATE,
    MAX_EPOCHS,
    PATIENCE,
    SEED,
    STRIDE_SAMPLES,
    WEIGHT_DECAY,
    WINDOW_SAMPLES,
)
from losses import CWLLoss
from models.cwl_ecg import CWLECG


def make_loader(signals, labels, shuffle=False):
    windows = np.lib.stride_tricks.sliding_window_view(labels, WINDOW_SAMPLES, axis=-1)
    windows = windows[..., ::STRIDE_SAMPLES, :]
    samples = torch.from_numpy(signals.reshape(-1, 1, WINDOW_SAMPLES))
    targets = torch.from_numpy(windows.reshape(-1, WINDOW_SAMPLES).astype(np.int64))
    dataset = TensorDataset(samples, targets)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)


@torch.no_grad()
def validation_loss(model, loader, loss_fn, device):
    model.eval()
    total = torch.zeros((), device=device)
    for signals, labels in loader:
        loss = loss_fn(model(signals.to(device)), labels.to(device))
        total += len(signals) * loss
    return total.item() / len(loader.dataset)


def fit(signals, labels, train_indices, val_indices, device):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    train_loader = make_loader(signals[train_indices], labels[train_indices], shuffle=True)
    val_loader = make_loader(signals[val_indices], labels[val_indices])
    model = CWLECG().to(device)
    loss_fn = CWLLoss().to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
    best_loss = math.inf
    best_state = None
    stale_epochs = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for batch, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch.to(device)), targets.to(device))
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        val_loss = validation_loss(model, val_loader, loss_fn, device)
        if not math.isfinite(val_loss):
            raise RuntimeError(f"Non-finite validation loss at epoch {epoch}.")
        scheduler.step(val_loss)
        print(f"Epoch {epoch:3d}  validation loss {val_loss:.5f}", flush=True)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {
                name: tensor.to("cpu", copy=True) for name, tensor in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs == PATIENCE:
                break

    model.load_state_dict(best_state)
    return model
