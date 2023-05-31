import argparse
from json import loads
from os import listdir
from os.path import exists, isdir, join
from shutil import rmtree
import albumentations as A
import cv2
from ultralytics import YOLO

from src.config import *
from src.data import MOT20Dataset, MOT20ExtDataset, MOT20Object
from src.data.preparing import (
    check_train_test_differs,
    extract_mot20_ext_test,
    restore_annotations,
    restore_dataset
)
from src.data.preparing import run as run_mot20_extracting
from src.tracker import run as run_tracker
from src.train.utils import get_experiments, get_model as get_saved_model
from src.config import MOT20_EXT_MEAN
from src.transforms import get_norm_transform, get_resize_transform


def get_model():
    df = get_experiments().sort_values('best_val_acc', ascending=False)
    best = df[df['datetime'] == '2023-05-16 21:04:15.317697']
    model = get_saved_model(best)
    threshold = 9
    return model, threshold


def prepare_image(mat, transform):
    return transform(image=cv2.cvtColor(mat, cv2.COLOR_BGR2RGB))[
        'image'].unsqueeze(0)


def get_model_predict(model, threshold, a, b):
    """Возвращает boolean - являются ли два объекта одинаковыми"""
    resize_transform = get_resize_transform(
        (MOT20_EXT_FIRST_AXIS_MEAN, MOT20_EXT_SECOND_AXIS_MEAN))
    norm_transform = get_norm_transform()
    transform = A.Compose([resize_transform, norm_transform])
    predict = model(
        prepare_image(a, transform),
        prepare_image(b, transform)
    )
    return predict < threshold


def parse(file_name):
    with open(file_name) as f:
        return loads(f.read())


def main(args):
    if (args.create_dataset):
        if (args.create_dataset == 'mot20'):
            mot20_path = join(DATA_PATH, 'MOT20')
            if not (exists(mot20_path) and isdir(mot20_path)):
                raise ValueError(
                    'MOT20 data directory is not exists. Load MOT20 dataset first')

            # run_mot20_extracting(data_path=DATA_PATH)
            # if (DEBUG):
            # rmtree(join(DATA_PATH, 'MOT20_ext'), )
            # extract_mot20_ext_test(data_path=DATA_PATH, proportion=0.25)
            check_train_test_differs(data_path=DATA_PATH)
            # restore_dataset(DATA_PATH)
            # restore_annotations(DATA_PATH)

    # if (args.track_video):
    if True:
        # path = args.track_video
        path = join(DATA_PATH, 'wisenet_dataset')
        video_path = join(path, 'video_sets', 'set_3')
        calibration_path = join(path, 'network_enviroment',
                                'camera_calibration', '1280_720')
        calibration_data = list(
            map(lambda f: (parse(join(calibration_path, f)), f), listdir(calibration_path)))
        calibration_data.sort(key=lambda x: x[1])
        detector = YOLO('yolov8n.pt')
        model, threshold = get_model()
        captures = list(map(lambda n: (cv2.VideoCapture(
            join(video_path, n)), n), listdir(video_path)))
        captures.sort(key=lambda x: x[1])
        run_tracker(captures, calibration_data, detector, lambda a,
                    b: get_model_predict(model, threshold, a, b))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--create-dataset',
        help="Создает датасет с указанным именем. Доступные значения: mot20"
    )
    parser.add_argument(
        '-tv', '--track-video',
        help='Запускает трекинг для видео по заданному пути'
    )
    args = parser.parse_args()
    main(args)
