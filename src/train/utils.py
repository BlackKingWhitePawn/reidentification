from datetime import datetime
from os.path import exists, join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import IMAGENET_MEAN, IMAGENET_STD, RESULTS_PATH


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
        optimizer: str,
        loss_name: str,
        dataset_config: str,
        val_losses: list[float],
        val_accuracies: list[float],
        # test_accuracy: float = None,
        gamma: float = -1,
        step_size: int = -1,
        config: dict = None
):
    """Сохраняет результат обучения в датафрейм
    ### Parameters:
    - model_name: str
    - datetime: datetime
    - epoch_count: int
    - lr: float
    - optimizer: str
    - loss_name: str
    - dataset_config: str
    - val_losses: list[float]
    - val_accuracies: list[float]
    - gamma: float
    - step_size: int
    - config: dict
    """
    file_path = join(RESULTS_PATH, 'experiments.csv')
    df = None
    if (not exists(file_path)):
        # TODO: указать типы
        df = pd.DataFrame(columns=[
            'model_name',
            'datetime',
            'epoch_count',
            'optimizer',
            'lr',
            'gamma',
            'step_size',
            'loss_name',
            'dataset_config'
        ])
        df.to_csv(file_path, sep=',', index=False)
    else:
        df = pd.read_csv(file_path)

    df = df.append(pd.DataFrame({
        'model_name': model_name,
        'datetime': datetime,
        'epoch_count': epoch_count,
        'optimizer': optimizer,
        'lr': lr,
        'gamma': gamma if (gamma > 0) else np.nan,
        'step_size': step_size if (step_size > 0) else np.nan,
        'loss_name': loss_name,
        'val_losses': ';'.join(map(str, val_losses)),
        'val_accuracies': ';'.join(map(str, val_accuracies)),
        'test': np.nan,
        'dataset_config': config['dataset_config'] if (config is not None) else dataset_config
    }, index=[0]))
    df.to_csv(file_path, sep=',', index=False)

    if (config is not None):
        config_path = join(RESULTS_PATH, 'configs.csv')
        df_config = None
        if (not exists(config_path)):
            # TODO: указать типы
            df_config = pd.DataFrame(columns=[
                'dataset_config',
                'dataset',
                'dataset_use',
                'train_proportion',
                'val_proportion',
                'test_proportion',
                'batch_size',
                'extra_values'
            ])
            df_config.to_csv(config_path, sep=',', index=False)
        else:
            df_config = pd.read_csv(config_path)

        if (config['dataset_config'] not in df_config['dataset_config'].unique()):
            config['extra_values'] = ';'.join([f'{k}={v}' for k, v in zip(
                config['extra_values'], config['extra_values'].values())])
            config_append = pd.DataFrame(config, index=[0])
            df_config = df_config.append(config_append)
            df_config.to_csv(config_path, sep=',', index=False)


def save_test_results():
    """Сохраняет в таблицу с экспериментами результат теста для соответствующей конфигурации"""
    pass
