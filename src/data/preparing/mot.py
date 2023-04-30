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

from os import listdir, mkdir
from os.path import exists, join, isdir
import pandas as pd


def save_objects(video_path: str, objects: list[any]) -> None:
    """Сохраняет список вырезанных объектов в соответствующие директории.
    Файлы вырезанных объектов именуются номером фрейма, из которого они вырезаны
    ### Parameters: 
    - video_path: str - путь до директории с видео в датасете MOT20_ext
    - objects: list
    """
    pass


def extract_objects_from_frame(frame: str, data: pd.DataFrame) -> list[any]:
    """Извлекает все объекты из кадра.
    Возвращает список 
    ### Parameters:
    - frame: str - путь до изображения, из которого вырезаются объекты
    - data: pandas.DataFrame - датафрейм с данными объектов
    """
    pass


def extract_video(mot20_video_path: str, mot20_video_ext_path: str) -> None:
    """Извлекает все объекты из видео
    ### Parameters: 
    - mot20_video_path: str - путь до директории с видео в датасете MOT20
    - mot20_video_ext_path: str - путь до директории с видео в датасете MOT20_ext
    """
    detections = get_dataframe(mot20_video_path, file_type='det')
    ground_truth = get_dataframe(mot20_video_path, file_type='gt')
    # выбираем объекты, которые стоит рассматривать и которые относятся к классу пешеходов или стоящих людей
    persons = ground_truth[((ground_truth['class'] == 1) | (
        ground_truth['class'] == 7)) & ground_truth['is_consider'] == 1]
    # для каждого объекта создаем директорию
    for id in persons['id'].unique():
        mkdir(join(mot20_video_ext_path, str(id)))
    # проходим по всем кадрам видео
    for frame in persons['frame'].unique():
        extract_objects_from_frame(
            join(mot20_video_path, ''), persons[persons['frame'] == frame])


def get_dataframe(video_path: str, file_type: str = 'det') -> pd.DataFrame:
    """
    Возвращает датафрейм с обнаружениями или ground truth для указанного видео
    ### Parameters
    - video_path: str - путь до директории с видео
    - file_type: str - det для файла с обнаружениями, gt для ground truth
    """
    df = None
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

    if (file_type == 'det'):
        df = pd.read_csv(
            join(video_path, 'det', 'det.txt'),
            names=DET_COLUMNS,
            dtype=DET_TYPES
        )
    else:
        df = pd.read_csv(
            join(video_path, 'gt', 'gt.txt'),
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
    mot20_ext_path = join(data_path, 'MOT20_ext')
    mot20_path = join(data_path, 'MOT20')
    # создание директорий
    if (not (exists(mot20_ext_path) and isdir(mot20_ext_path))):
        # создаем основную директорию
        mkdir(mot20_ext_path)
        # мы используем данные для трейна, так как мы не будем обучаться на тесте
        mkdir(join(mot20_ext_path, 'train'))
        # проходим по всем видео в исходной
        for video_id in listdir(join(mot20_path, 'train')):
            # сохраняем пути до видео в исходной и в новой директориях
            current_path = join(
                join(data_path, 'MOT20'), 'train', video_id)
            current_path_ext = join(mot20_ext_path, 'train', video_id)
            mkdir(current_path_ext)
            extract_video(current_path, current_path_ext)