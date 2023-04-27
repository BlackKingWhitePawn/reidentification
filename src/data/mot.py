from torch.utils.data import Dataset
import os
from os.path import join
from tqdm import tqdm
import cv2
import pandas as pd
import numpy as np
from data.person import Person


class MOT20Dataset(Dataset):
    """Создает объект типа Dataset, загружающий данные датасета MOT20"""

    def __init__(self, path, transform=None) -> None:
        super().__init__()
        self.paths = []
        source_dir = join(path, 'MOT20')
        if (os.path.exists(source_dir) and os.path.isdir(source_dir)):
            for data_type in ['test', 'train']:
                for data in os.listdir(join(source_dir, data_type)):
                    for file_name in os.listdir(join(source_dir, data_type, data, 'img1')):
                        if (not file_name.split('.')[1] == 'jpg'):
                            continue
                        self.paths.append(join(source_dir, file_name))

        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image_path = self.paths[idx]
        image = cv2.imread(image_path)
        if self.transform:
            image = self.transform(image=image)['image']
        # return image, 0 if (image_path.split('/')[-2] == 'dirty') else


class MOT20Object:
    """
    Представляет элемент датасета MOT20.
    Содержит путь до изображения и список меток объектов, содержащихся на нем
    """

    def __init__(self, path: str, labels: list[str]) -> None:
        self.path = path
        self.labels = labels

    def extract_objects() -> list[Person]:
        pass

    @staticmethod
    def get_data_annotations(dataset_path_prefix):
        annotations = None
        with open(join(dataset_path_prefix, 'gt', 'gt.txt')) as f:
            annotations = pd.DataFrame(
                list(
                    map(lambda l: np.array(
                        list(
                            map(lambda x: float(x), l.split(',')))), f.readlines()))
            )
            annotations.columns = ['frame', 'id', 'bb_left', 'bb_top',
                                   'bb_width', 'bb_height', 'flag?', 'class', 'conf']
            annotations.astype({
                'frame': int,
                'id': int,
                'bb_left': float,
                'bb_top': float,
                'bb_width': float,
                'bb_height': float,
                'flag?': int,
                'class': int,
                'conf': float
            })

        return annotations

    # annotations = get_data_annotations(MOT2001_TRAIN_PATH)
    # annotations
