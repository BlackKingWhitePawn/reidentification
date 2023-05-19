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
from os.path import exists, isdir, join
from random import sample
from shutil import move, copytree, rmtree


import pandas as pd
from PIL import Image
from tqdm import tqdm

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


def __extract_objects_from_frame(video_path: str, frame: str, data: pd.DataFrame) -> None:
    """Извлекает все объекты из кадра. Сохраняет список вырезанных объектов в соответствующие директории
    ### Parameters:
    - video_path: str - путь до директории с текущим видео
    - frame: str - путь до изображения, из которого вырезаются объекты
    - data: pandas.DataFrame - датафрейм с данными объектов на данном изображении
    """
    current_image = Image.open(frame)
    objects_to_crop = data.to_dict('records')
    for obj in objects_to_crop:
        x, y, w, h = obj['bb_left'], obj['bb_top'], obj['bb_width'], obj['bb_height']
        img = current_image.crop(box=(x, y, x + w, y + h))
        img.save(join(video_path, str(
            int(obj['id'])), __get_file_name_by_id(obj['frame'])))


def __extract_video(mot20_video_path: str, mot20_video_ext_path: str) -> None:
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
    for frame in tqdm(persons['frame'].unique()):
        __extract_objects_from_frame(
            mot20_video_ext_path,
            join(
                mot20_video_path, 'img1', f'{str.zfill(str(frame), 6)}.jpg'),
            persons[persons['frame'] == frame]
        )
        # save_objects(mot20_video_ext_path, objects)


def get_dataframe(video_path: str, file_type: str = 'det') -> pd.DataFrame:
    """
    Возвращает датафрейм с обнаружениями или ground truth для указанного видео
    ### Parameters
    - video_path: str - путь до директории с видео
    - file_type: str - det для файла с обнаружениями, gt для ground truth
    """
    df = None

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
            __extract_video(current_path, current_path_ext)


def extract_mot20_ext_test(data_path: str, proportion: float) -> None:
    """
    Выполняет преобразование тестовой части датасета MOT20_ext.
    Случайным образом переносит в тест указанную часть объектов из каждого видео.
    Это делается для того, чтобы модели не могли обучиться на людях, которых они видели на обучении
    ### Parameters
        - data_path: str - путь до директории с датасетами
        - proportion: float - число объектов, которые нужно перенести в тест
    """
    mot20_ext_path = join(data_path, 'MOT20_ext')
    # создание директорий
    if (not (exists(mot20_ext_path) and isdir(mot20_ext_path))):
        raise ValueError(
            'MOT20_ext data directory is not exists. Create MOT20_Ext dataset first by calling run function')

    if (not exists(join(mot20_ext_path, 'test'))):
        mkdir(join(mot20_ext_path, 'test'))

    for video_id in listdir(join(mot20_ext_path, 'train')):
        current_path_ext = join(mot20_ext_path, 'train', video_id)
        new_path_ext = join(mot20_ext_path, 'test', video_id)
        if (not exists(new_path_ext)):
            mkdir(new_path_ext)
        # скопировать аннотации
        copytree(join(current_path_ext, 'det'), join(new_path_ext, 'det'))
        copytree(join(current_path_ext, 'gt'), join(new_path_ext, 'gt'))
        # выбрать объекты которые переносятся в тест
        objects = listdir(current_path_ext)
        to_extract = sample(objects, round(len(objects) * proportion))
        to_extract = list(map(int, set(to_extract) - set(['gt', 'det'])))
        # перенести все аннтоации по нужным объектам в новый датафрейм
        gt = get_dataframe(current_path_ext, 'gt')
        mask_test = gt['id'].apply(lambda id: id in to_extract)
        mask_train = gt['id'].apply(lambda id: id not in to_extract)
        train_gt = gt[mask_train]
        test_gt = gt[mask_test]
        train_gt.to_csv(join(current_path_ext, 'gt', 'gt.txt'),
                        index=False, header=False)
        test_gt.to_csv(join(new_path_ext, 'gt', 'gt.txt'),
                       index=False, header=False)
        # переместить объекты
        for object_id in to_extract:
            move(join(current_path_ext, f'{object_id}'), new_path_ext)


def restore_dataset(data_path: str):
    """Возвращает тест в трейн"""
    train_path = join(data_path, 'MOT20_ext', 'train')
    test_path = join(data_path, 'MOT20_ext', 'test')
    for video_id in listdir(test_path):
        for object_id in listdir(join(test_path, video_id)):
            if not (object_id == 'gt' or object_id == 'det'):
                move(join(test_path, video_id, object_id),
                     join(train_path, video_id))


def restore_annotations(data_path: str):
    original = join(data_path, 'MOT20', 'train')
    train_path = join(data_path, 'MOT20_ext', 'train')
    test_path = join(data_path, 'MOT20_ext', 'test')
    for video_id in listdir(original):
        # удалить переписанные аннотации
        rmtree(join(train_path, video_id, 'det'))
        rmtree(join(train_path, video_id, 'gt'))
        rmtree(join(test_path, video_id, 'det'))
        rmtree(join(test_path, video_id, 'gt'))
        # скопировать из оригинального датасета
        copytree(join(original, video_id, 'det'),
                 join(train_path, video_id, 'det'))
        copytree(join(original, video_id, 'gt'),
                 join(train_path, video_id, 'gt'))


def check_train_test_differs(data_path: str):
    """Проверяет, что никакие тестовые данные не содержатся в трейне"""
    train_path = join(data_path, 'MOT20_ext', 'train')
    test_path = join(data_path, 'MOT20_ext', 'test')
    for video_id in listdir(train_path):
        train = listdir(join(train_path, video_id))
        test = listdir(join(test_path, video_id))
        res = set(train) & set(test)
        res = res - set(['gt', 'det'])
        if (bool(res)):
            raise ValueError(
                f'Train and test has similar element in {video_id}')
