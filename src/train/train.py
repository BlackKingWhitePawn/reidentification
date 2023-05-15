import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils import save_train_results


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


def train_siamese(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: Optimizer,
    lr: float,
    criterion,
    epoch_count: int,
    threshold: float,
    scheduler: None = None,
    device: torch.device = torch.device('cpu'),
    config: dict = None,
):
    """Обучает сиамскую модель с выбранными параметрами.
    Сохраняет результаты обучения в таблицу.
    ### Parameters:
    - model: Module - обучаемая модель
    - train_loader: DataLoader - трейн лоадер
    - val_loader: DataLoader - валидационный лоадер
    - optimizer: Optimizer - оптимизационный метод
    - criterion - лосс-функция
    - epoch_count: int - кол-во эпох. нумерация начинается с нулевой
    - scheduler - планировщик lr
    - threshold: float - порог, разбивающий предсказания для батча. 
        Все предсказание предварительно нормализуется в границах [0;1] 
    - device: device - устройство
    - config: dict - конфигурация датасета
    """
    losses_train = []
    accuracies_train = []
    losses_val = []
    accuracies_val = []
    best_val_accuracy = 0
    best_val_loss = 1e10
    dt = None

    for epoch in range(epoch_count):
        print('Epoch {}/{}:'.format(epoch, epoch_count - 1), flush=True)
        for phase in ['train', 'val']:
            if (phase == 'train'):
                dataloader = train_loader
                if (scheduler is not None):
                    scheduler.step()
                model.train()
            else:
                dataloader = val_loader
                model.eval()

            running_loss = 0.
            running_acc = 0.

            for (x1, x2, y) in tqdm(dataloader):
                x1, x2, y = x1.to(device), x2.to(device), y.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    distance = model(x1, x2)
                    loss = criterion(distance, y)
                    d = distance.clone().reshape(-1)
                    # нормализация к [0;1]
                    d_min, _ = torch.min(d, dim=0)
                    d_max, _ = torch.max(d, dim=0)
                    d = (d - d_min) / (d_max - d_min)
                    d[d <= threshold] = 0
                    d[d > threshold] = 1

                    if (phase == 'train'):
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item()
                running_acc += torch.eq(d, y).float().mean()

            epoch_loss = running_loss / len(dataloader)
            epoch_acc = running_acc / len(dataloader)
            if phase == 'val':
                losses_val.append(epoch_loss)
                accuracies_val.append(epoch_acc)
            else:
                losses_train.append(epoch_loss)
                accuracies_train.append(epoch_acc)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc), flush=True)

            if phase == 'val' and (best_val_accuracy < epoch_acc
                                   or (best_val_accuracy == epoch_acc and epoch_loss < best_val_loss)):
                best_val_accuracy = epoch_acc
                best_val_loss = epoch_loss
                dt = datetime.now()
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_loss,
                    'model_name': model.name,
                }, f'./models/{model.name}_{dt.strftime("%d.%m_%H:%M")}.pth')
                print(f'Model saved at {model.name}.pth')

    save_train_results(
        model_name=model.name,
        datetime=dt,
        epoch_count=epoch_count,
        lr=lr,
        optimizer=str(optimizer).split(' ')[0],
        loss_name=str(criterion)[:-2],
        val_losses=losses_val,
        val_accuracies=accuracies_val,
        gamma=scheduler.gamma if (scheduler) else -1,
        step_size=scheduler.step_size if (scheduler) else -1,
        config=config,
        extra_parameters={
            'threshold': threshold
        }
    )

    return model, {
        'train': (losses_train, accuracies_train),
        'val': (losses_val, accuracies_val)
    }
