import sys

import numpy as np
import torch.nn as nn
from tqdm.auto import tqdm
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import torch


def train_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: Optimizer,
    criterion,
    device: torch.device,
    return_losses=False,
):
    """Выполняет одну эпоху обучения переданной модели"""
    model = model.to(device).train()
    total_loss = 0
    num_batches = 0
    all_losses = []
    total_predictions = np.array([])  # .reshape((0, ))
    total_labels = np.array([])  # .reshape((0, ))
    with tqdm(total=len(data_loader), file=sys.stdout) as prbar:
        for inputs, labels in data_loader:
            # Move Batch to GPU
            inputs = inputs.to(device)
            labels = labels.to(device)
            predicted = model(inputs)
            loss = criterion(predicted, labels)
            # Update weights
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # Update descirption for tqdm
            accuracy = (predicted.argmax(1) == labels).float().mean()
            prbar.set_description(
                f"Loss: {round(loss.item(), 4)} "
                f"Accuracy: {round(accuracy.item() * 100, 4)}"
            )
            prbar.update(1)
            total_loss += loss.item()
            total_predictions = np.append(
                total_predictions, predicted.argmax(1).cpu().detach().numpy())
            total_labels = np.append(
                total_labels, labels.cpu().detach().numpy())
            num_batches += 1
            all_losses.append(loss.detach().item())
    metrics = {"loss": total_loss / num_batches}
    metrics.update({"accuracy": (total_predictions == total_labels).mean()})
    if return_losses:
        return metrics, all_losses
    else:
        return metrics


def validate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion,
    device="cuda:0"
):
    """Выполняет этап валидации переданной модели"""
    model = model.eval()
    total_loss = 0
    num_batches = 0
    total_predictions = np.array([])
    total_labels = np.array([])
    with tqdm(total=len(data_loader), file=sys.stdout) as prbar:
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            predicted = model(inputs)
            loss = criterion(predicted, labels)
            accuracy = (predicted.argmax(1) == labels).float().mean()
            prbar.set_description(
                f"Loss: {round(loss.item(), 4)} "
                f"Accuracy: {round(accuracy.item() * 100, 4)}"
            )
            prbar.update(1)
            total_loss += loss.item()
            total_predictions = np.append(
                total_predictions, predicted.argmax(1).cpu().detach().numpy())
            total_labels = np.append(
                total_labels, labels.cpu().detach().numpy())
            num_batches += 1
    metrics = {"loss": total_loss / num_batches}
    metrics.update({"accuracy": (total_predictions == total_labels).mean()})
    return metrics


def fit(
    model: nn.Module,
    epochs_count: int,
    train_data_loader: DataLoader,
    validation_data_loader: DataLoader,
    optimizer: Optimizer,
    criterion,
    device: torch.device
):
    """Выполняет обучение модели"""
    all_train_losses = []
    epoch_train_losses = []
    epoch_eval_losses = []
    for epoch in range(epochs_count):
        # Train step
        print(f"Train Epoch: {epoch}")
        train_metrics, one_epoch_train_losses = train_epoch(
            model=model,
            data_loader=train_data_loader,
            optimizer=optimizer,
            return_losses=True,
            criterion=criterion,
            device=device
        )
        # Save Train losses
        all_train_losses.extend(one_epoch_train_losses)
        epoch_train_losses.append(train_metrics["loss"])
        # Eval step
        print(f"Validation Epoch: {epoch}")
        with torch.no_grad():
            validation_metrics = validate(
                model=model,
                data_loader=validation_data_loader,
                criterion=criterion
            )
        # Save eval losses
        epoch_eval_losses.append(validation_metrics["loss"])
