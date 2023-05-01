"""
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

from os import listdir as __listdir
from os import mkdir as __mkdir
from os.path import exists as __exists
from os.path import isdir as __isdir
from os.path import join as __join

import pandas as __pd
from PIL import Image as __Image
from tqdm import tqdm as __tqdm

DET_COLUMNS = ['frame', 'id', 'bb_left', 'bb_top',
               'bb_width', 'bb_height', 'confidence', 'x', 'y', 'z']
DET_TYPES = {
    'frame': int,
    'id': int,
    'bb_left': int,
    'bb_top': int,
    'bb_width': int,
    'bb_height': int,
    'confidence': int,
    'x': int,
    'y': int,
    'z': int
}

GT_COLUMNS = ['frame', 'id', 'bb_left', 'bb_top',
              'bb_width', 'bb_height', 'is_consider', 'class', 'visibility']

GT_TYPES = {
    'frame': int,
    'id': int,
    'bb_left': int,
    'bb_top': int,
    'bb_width': int,
    'bb_height': int,
    'is_consider': int,
    'class': int,
    'visibility': float
}


def __get_file_name_by_id(frame_id: int) -> str:
    return f'{str.zfill(str(frame_id), 6)}.jpg'


def __extract_objects_from_frame(video_path: str, frame: str, data: __pd.DataFrame) -> None:
    """Извлекает все объекты из кадра. Сохраняет список вырезанных объектов в соответствующие директории
    ### Parameters:
    - video_path: str - путь до директории с текущим видео
    - frame: str - путь до изображения, из которого вырезаются объекты
    - data: pandas.DataFrame - датафрейм с данными объектов на данном изображении
    """
    current_image = __Image.open(frame)
    objects_to_crop = data.to_dict('records')
    for obj in objects_to_crop:
        x, y, w, h = obj['bb_left'], obj['bb_top'], obj['bb_width'], obj['bb_height']
        img = current_image.crop(box=(x, y, x + w, y + h))
        img.save(__join(video_path, str(
            int(obj['id'])), __get_file_name_by_id(obj['frame'])))


def __extract_video(mot20_video_path: str, mot20_video_ext_path: str) -> None:
    """Извлекает все объекты из видео
    ### Parameters: 
    - mot20_video_path: str - путь до директории с видео в датасете MOT20
    - mot20_video_ext_path: str - путь до директории с видео в датасете MOT20_ext
    """
    detections = __get_dataframe(mot20_video_path, file_type='det')
    ground_truth = __get_dataframe(mot20_video_path, file_type='gt')
    # выбираем объекты, которые стоит рассматривать и которые относятся к классу пешеходов или стоящих людей
    persons = ground_truth[((ground_truth['class'] == 1) | (
        ground_truth['class'] == 7)) & ground_truth['is_consider'] == 1]
    # для каждого объекта создаем директорию
    for id in persons['id'].unique():
        __mkdir(__join(mot20_video_ext_path, str(id)))
    # проходим по всем кадрам видео
    for frame in __tqdm(persons['frame'].unique()):
        __extract_objects_from_frame(
            mot20_video_ext_path,
            __join(
                mot20_video_path, 'img1', f'{str.zfill(str(frame), 6)}.jpg'),
            persons[persons['frame'] == frame]
        )
        # save_objects(mot20_video_ext_path, objects)


def __get_dataframe(video_path: str, file_type: str = 'det') -> __pd.DataFrame:
    """
    Возвращает датафрейм с обнаружениями или ground truth для указанного видео
    ### Parameters
    - video_path: str - путь до директории с видео
    - file_type: str - det для файла с обнаружениями, gt для ground truth
    """
    df = None

    if (file_type == 'det'):
        df = __pd.read_csv(
            __join(video_path, 'det', 'det.txt'),
            names=DET_COLUMNS,
            dtype=DET_TYPES
        )
    else:
        df = __pd.read_csv(
            __join(video_path, 'gt', 'gt.txt'),
            names=GT_COLUMNS,
            dtype=GT_TYPES
        )

    return df


def run(data_path: str) -> None:
    """
    Выполняет преобразование датасета MOT20
    ### Parameters
    - data_path: str - путь до директории с датасетами
    """
    mot20_ext_path = __join(data_path, 'MOT20_ext')
    mot20_path = __join(data_path, 'MOT20')
    # создание директорий
    if (not (__exists(mot20_ext_path) and __isdir(mot20_ext_path))):
        # создаем основную директорию
        __mkdir(mot20_ext_path)
        # мы используем данные для трейна, так как мы не будем обучаться на тесте
        __mkdir(__join(mot20_ext_path, 'train'))
        # проходим по всем видео в исходной
        for video_id in __listdir(__join(mot20_path, 'train')):
            # сохраняем пути до видео в исходной и в новой директориях
            current_path = __join(
                __join(data_path, 'MOT20'), 'train', video_id)
            current_path_ext = __join(mot20_ext_path, 'train', video_id)
            __mkdir(current_path_ext)
            __extract_video(current_path, current_path_ext)
