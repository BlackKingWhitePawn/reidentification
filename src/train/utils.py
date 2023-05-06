import matplotlib as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def display_images(img_tensors: tuple, label: int):
    """Отрисовывает переданные изображения. Принимает нормализованные изображения в виде тензоров"""
    image1 = img_tensors[0].permute(1, 2, 0).numpy()
    image2 = img_tensors[1].permute(1, 2, 0).numpy()
    plt.imshow(image1.clip(0, 1))
    plt.imshow(image2.clip(0, 1))
    plt.title('similar' if (label == 1) else 'different')
    plt.show()


def get_statistics(loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
    """Вычисляет среднее и стандартное отклонение для переданых данных.
    ### Parameters:
    - loader: torch.utils.data.DataLoader - даталоадер, загружающий Mot20ExtDataset
    """
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
