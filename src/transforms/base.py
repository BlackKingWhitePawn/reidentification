import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
from src.config import IMAGENET_MEAN, IMAGENET_STD


def get_resize_transform(size: tuple[int, int]) -> A.Sequential:
    """Возвращает преобразование изменения размера"""
    return A.Sequential([
        A.Resize(*size),
    ])


def get_norm_transform(mean: list[int] = IMAGENET_MEAN, std: list[int] = IMAGENET_STD) -> A.Sequential:
    """Возвращает преобразование нормализации и приведения к тензору"""
    return A.Sequential([
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])
