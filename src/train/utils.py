import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

from src.config import IMAGENET_MEAN, IMAGENET_STD


def display_images(img_tensors: tuple[torch.Tensor, torch.Tensor], label: int, mean: list[int] = IMAGENET_MEAN, std: list[int] = IMAGENET_STD):
    """Отрисовывает переданные изображения. Принимает нормализованные изображения в виде тензоров"""
    image1 = std * img_tensors[0].permute(1, 2, 0).numpy() + mean
    image2 = std * img_tensors[1].permute(1, 2, 0).numpy() + mean
    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(image1.clip(0, 1))
    axarr[1].imshow(image2.clip(0, 1))
    axarr[0].text(0.0, 1.0, 'similar' if (label == 0) else 'different',
                  bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0))


def display_batch(batch, mean: list[int] = IMAGENET_MEAN, std: list[int] = IMAGENET_STD):
    """Отрисовывает батч пар"""
    # TODO: Исправить отрисовку для множества осей
    fig, axs = plt.subplots(nrows=len(batch[0]), ncols=2)
    print(len(batch[0]))
    for x1, x2, y, i in tqdm(zip(batch[0], batch[1], batch[2], range(len(batch[0])))):
        image1 = std * x1.permute(1, 2, 0).numpy() + mean
        image2 = std * x2.permute(1, 2, 0).numpy() + mean
        axs[i, 0].imshow(image1.clip(0, 1))
        axs[i, 1].imshow(image2.clip(0, 1))
        axs[i, 0].text(0.0, 1.0, 'similar' if (y == 0) else 'different',
                       bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0))


def get_statistics(loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
    """Вычисляет среднее и стандартное отклонение для переданых данных.
    ### Parameters:
    - loader: torch.utils.data.DataLoader - даталоадер, загружающий Mot20ExtDataset
    """
    # TODO: исправить подсчет для формата с tuple
    mean = 0.
    std = 0.
    nb_samples = 0.
    for data in tqdm(loader):
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    return (mean, std)


def save_train_results(
        model_name: str,
        datetime: datetime,
        epoch_count: int,
        lr: float,
        loss_name: str,
        dataset: str,
        gamma: float = -1,
        step_size: int = -1
):
    """Сохраняет результат обучения в датафрейм"""
    pass
