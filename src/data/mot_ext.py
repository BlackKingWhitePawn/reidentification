"""
Содержит классы для загрузки данных MOT20_ext, преобразованных из MOT20 dataset
MOT20_ext data format:
test:
|
|- <video_id> // директория, содержащая объекты, вырезанные из соответствующего видео
    |
    |- <object_id> // директория, содержащая изображения вырезанных объектов и файл описания
        |
        |- det
        |   |
        |   |- det.txt // файл описания, хранящий строки в формате <frame number>, <object id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <confidence>, <x>, <y>, <z>
        |
        |- img1 // директория с изображениями
            |
            .
            .
            .
            |- <frame_id>.jpg - вырезанная из кадра frame_id область с объектом object_id

"""

from torch.utils.data import Dataset
from os.path import join
from os import listdir
from tqdm import tqdm
import cv2
import pandas as pd
import numpy as np


class MOT20ExtDataset(Dataset):
    """
    Создает объект типа Dataset, загружающий данные преобразованного датасета MOT20_ext
    Возвращает пары изображений и метку: 1, если на изображении один и тот же объект, иначе 0 
    """

    def __init__(self, vid_path: str) -> None:
        """
        ### Parameters:
        - vid_path: str

            путь до директории, содержащей объекты, вырезанные из видео MOT20
        """
        super(MOT20ExtDataset).__init__()
        self.vid_path = vid_path

    def __len__(self) -> int:
        object_dirs = listdir(join('data', 'mot20_ext', self.vid_path))
        object_frames_counts = [len(x) for x in object_dirs]
        return
        # for object_dir in listdir(join('data', 'mot20_ext', self.vid_path)):
        #     print(object_dir)

    def __getitem__(self, idx) -> tuple[cv2.Mat, cv2.Mat, int]:
        # image_path = self.paths[idx]
        image = cv2.imread(image_path)
        return (image, image, 0)
        # if self.transform:
        #     image = self.transform(image=image)['image']


MOT20ExtDataset()
