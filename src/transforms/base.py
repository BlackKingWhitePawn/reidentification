import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2


def get_resize_transform(size: tuple[int, int]) -> A.Sequential:
    """Возвращает преобразование изменения размера"""
    return A.Sequential([
        A.Resize(*size),
    ])


def get_norm_transform(mean: list[int] = [0.485, 0.456, 0.406], std: list[int] = [0.229, 0.224, 0.225]) -> A.Sequential:
    """Возвращает преобразование нормализации и приведения к тензору"""
    return A.Sequential([
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])
