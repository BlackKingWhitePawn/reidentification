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
from .utils import get_possible_tuples_count, split_to_continuous_segments


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
            negative_proportion: float = 0.5
    ) -> None:
        """
        ### Parameters:
        - video_path: str - путь до директории с видео датасета МОТ20_ехт. Ожидается, что в директории находятся файлы описаний и ground truth
        - transform - применяемые аугментации
        - visibility_threshold: float - порог видимости (поле visibility) объекта, используемого в обучении
        - frame_distance: int | list[int] | tuple[int, int] - допустимое расстояние между кадрами, объекты из которых используются в обучении. Если переданы два числа в виде начального и конечного значений - конечное включается 
        - negative_proportion: float - доля объектов, значение метки для которых 0
        """
        super(MOT20ExtDataset).__init__()
        self.video_path = video_path
        self.visibility_threshold = visibility_threshold
        self._check_distance_correct(frame_distance)
        self.frame_distance = frame_distance
        self.detections = get_dataframe(video_path, file_type='det')
        df = get_dataframe(video_path, file_type='gt')
        # берем объекты которые стоит учитывать при обучении
        df = df[df['is_consider'] == 1]
        # выбираем с видимостью выше заданной
        df = df[df['visibility'] >= visibility_threshold]
        self.ground_truth = df
        # формируем словарь, используемый для длины и индексации
        self._objetcs_pairs_dict = self._get_pairs_dict()
        # заранее рассчитываем длину датасета
        self._len = self._calc_len()

    def _check_distance_correct(self, distance: int) -> None:
        """Проверяет корректность типов и значений для расстояния"""
        if (type(distance) == int):
            if (distance < 0):
                raise ValueError(
                    'Distance between frames must be non negative integer')
        elif (type(distance) == list):
            for d in distance:
                if (not type(d) == int):
                    raise TypeError(
                        'Each distance value must be non negative integer')
                if (distance < 0):
                    raise ValueError(
                        'Distance between frames must be non negative integer')
        elif (type(distance) == tuple):
            start, end = distance
            if (not (type(start) == int and type(end) == int)):
                raise TypeError(
                    'Each distance value must be non negative integer')
            if (start < 1):
                raise ValueError(
                    'Start index of distance must be non negative')
            if (end < start):
                raise ValueError(
                    'End index of distance must be bigger then start')
        else:
            raise TypeError(
                'Distance argument must be integre or list of integres or tuple of two integres')

    def _get_pairs_dict(self) -> dict[int, int | dict[int, int]]:
        """Возвращает словарь, содержащий количество возможных пар для каждого объекта. В случае нескольких d набор пар представлен как словари"""
        objects_lens = {}
        for object_id in sorted(self.ground_truth['id'].unique()):
            object_frames = sorted(
                self.ground_truth[self.ground_truth['id'] == object_id]['frame'].values)
            segments = split_to_continuous_segments(object_frames)
            if (type(self.frame_distance) == int):
                objects_lens[object_id] = get_possible_tuples_count(
                    self.frame_distance, segments)
            elif (type(self.frame_distance) == list):
                objects_lens[object_id] = {}
                for d in self.frame_distance:
                    objects_lens[object_id][d] = get_possible_tuples_count(
                        d, segments)
            else:
                start_d, end_d = self.frame_distance
                objects_lens[object_id] = {}
                for d in range(start_d, end_d + 1):
                    objects_lens[object_id][d] = get_possible_tuples_count(
                        d, segments)

        return objects_lens

    def _calc_len(self) -> int:
        """Рассчитывает длину датасета"""
        count = 0
        for v in self._objetcs_pairs_dict.values():
            if (type(v) == int):
                count += v
            else:
                count += sum(v.values())

        return count

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> tuple[cv2.Mat, cv2.Mat, int]:
        """Возвращает два изображения и метку: 1, если на изображении один и тот же объект, иначе 0
        Нумерация начинается с 0. Объекты в датасете хранятся в порядке:
        - сначала пары <объект;объект>, затем <объект;другой_объект>
        - пары <объект;объект> отсортированы по возрастанию id
        - пары для одного объекта отсортированы по возрастанию frame_id первого кадра пары
        """
        current_id = -1
        previous_pairs_count = 0
        for id, pairs in self._objetcs_pairs_dict.items():
            # считаем, сколько пар есть у данного объекта
            if (type(pairs) == int):
                previous_pairs_count += pairs
            else:
                previous_pairs_count = sum(pairs.values())

            # если индекс больше, чем пар у текущего объекта - берем следующий
            if (idx >= previous_pairs_count):
                continue
            else:
                #  ищем среди пар данного объекта
                iidx = idx - previous_pairs_count
                if (type(pairs) == int):
                    pass
                else:
                    for d, d_pairs in pairs:
                        pass
