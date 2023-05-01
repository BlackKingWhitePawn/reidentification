"""
Содержит классы для загрузки данных MOT20_ext, преобразованных из MOT20 dataset
MOT20_ext data format:
train:
|
|- <video_id> // директория, содержащая объекты, вырезанные из соответствующего видео
    |
    |- det
    |   |
    |   |- det.txt // файл описания детекций, хранящий строки в формате <frame number>, <object id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <confidence>, <x>, <y>, <z>
    |
    |- gt
    |   |
    |   |- gt.txt // файл описания ground truth, хранящий строки в формате <frame number>, <object id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <is_consider>, <class>, <visibility>
    |
    |- <object_id> // директория, содержащая изображения вырезанных объектов и файл описания
        |
        |
        .
        .
        .
        |- <frame_id>.jpg - вырезанная из кадра frame_id область с объектом object_id
"""

from os import listdir
from os.path import join

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset as Dataset
from tqdm import tqdm as tqdm

from .preparing.mot import get_dataframe


class MOT20ExtDataset(Dataset):
    """
    Создает объект типа Dataset, загружающий данные преобразованного датасета MOT20_ext
    Возвращает пары изображений и метку: 1, если на изображении один и тот же объект, иначе 0 
    """

    def __init__(
        self,
            video_path: str,
            transform=None,
            visibility_threshold: float = 1,
            frame_distance: int | list[int] | tuple[int, int] = 1,
    ) -> None:
        """
        ### Parameters:
        - video_path: str - путь до директории с видео датасета МОТ20_ехт. Ожидается, что в директории находятся файлы описаний и ground truth
        - transform - применяемые аугментации
        - visibility_threshold: float - порог видимости (поле visibility) объекта, используемого в обучении
        - frame_distance: int | list[int] | tuple[int, int] - допустимое расстояние между кадрами, объекты из которых используются в обучении
        """
        super(MOT20ExtDataset).__init__()
        self.video_path = video_path
        self.visibility_threshold = visibility_threshold
        self.frame_distance = frame_distance
        self.detections = get_dataframe(video_path, file_type='det')
        df = get_dataframe(video_path, file_type='gt')
        # берем объекты которые стоит учитывать при обучении
        df = df[df['is_consider'] == 1]
        # выбираем с видимостью выше заданной
        df = df[df['visibility'] >= visibility_threshold]
        self.ground_truth = df

    def __len__(self) -> int:
        if (type(self.frame_distance) == int):
            if (self.frame_distance < 1):
                raise ValueError(
                    'Distance between frames must be positive integer')

    def __getitem__(self, idx: int) -> tuple[cv2.Mat, cv2.Mat, int]:
        """Возвращает два изображения и метку: 1, если на изображении один и тот же объект, иначе 0"""
        pass
