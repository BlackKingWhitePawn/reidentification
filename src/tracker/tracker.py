from json import loads
from os import listdir
from os.path import join

import albumentations as A
import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
from torch.utils.data import DataLoader, random_split
from ultralytics import YOLO

from src.config import (DATA_PATH, MOT20_EXT_FIRST_AXIS_MEAN,
                        MOT20_EXT_SECOND_AXIS_MEAN)
from src.tracker import show_detections, show_roi_detection
from src.train import get_config, get_dataset, test_siamese
from src.train.utils import (draw_reid_predict, get_binary_accuracy,
                             get_config, get_experiments, get_model)
from src.transforms import get_norm_transform, get_resize_transform

path = join(DATA_PATH, 'wisenet_dataset')
listdir(path)


def get_iou(box1, box2):
    x_11 = box1[0]
    x_21 = x_11 + box1[2]
    y_11 = box1[1]
    y_21 = y_11 + box1[3]
    x_12 = float(box2[0])
    x_22 = x_12 + float(box2[2])
    y_12 = float(box2[1])
    y_22 = y_12 + float(box2[3])
    S_overlap = max(0, min(x_21, x_22) - max(x_11, x_12)) * \
        max(0, min(y_21, y_22) - max(y_11, y_12))
    S_union = box1[3] * box1[2] + box2[3] * box2[2] - S_overlap
    return S_overlap / S_union


def get_model():
    df = get_experiments().sort_values('best_val_acc', ascending=False)
    best = df[df['datetime'] == '2023-05-16 21:04:15.317697']
    model = get_model(best)
    threshold = 9
    return model, threshold


def get_model_predict(a, b):
    """Возвращает boolean - являются ли два объекта одинаковыми"""
    model, threshold = get_model()
    predict = model(a, b)
    return predict < threshold


def crop_from_frame(frame: cv2.Mat, box: tuple[float, float, float, float]):
    x, y, w, h = box
    return frame[y:y+h, x:x+w]


def run(camera_captures, camera_calibration):
    """Запускает отслеживание на перечисленных камерах"""
    # for cap, calibration, name in camera_data:
    # cap.release()

    videoset_path = join(path, 'video_sets', 'set_1')
    videos = list(map(lambda n: (cv2.VideoCapture(
        join(videoset_path, n)), n), listdir(videoset_path)))
    videos.sort(key=lambda x: x[1])
    camera_data = list(map(lambda x: (x[0][0], x[1][0], x[1][1].split('.')[
                       0]), zip(videos, calibration_data)))
    # camera_data[2][0].read()

    trackers = {}

    # создаем трекеры для каждой камеры
    for cap, calibration, name in camera_data:
        trackers[name] = DeepSort(max_age=30)

    frame_index = 0

    while True:
        # словарь текущих детекций для каждой камеры
        detections = {}
        # для каждого кадра получаем список детекций и обновляем трекеры
        for cap, calibration, name in camera_data:
            frame = cap.read()[1]
            res = detector.predict(frame, imgsz=1280, classes=[0])
            bbs = handle_boxes(pd.DataFrame(
                res[0].boxes.data.cpu(),
                columns=['x1', 'y1', 'x2', 'y2', 'confidence', 'class']
            ))
            # сохраненине детекции
            detections[name] = bbs
            # обновление трека сортом
            tracks = trackers[name].update_tracks(
                raw_detections=bbs, frame=frame)

        # создаем словарь всех регионов интереса
        match_roi = {}
        # проходим по всем камерам
        for cap, calibration, name in camera_data:
            # проходим по всем регионам интересов
            for camera_roi in calibration['regionsOfInterest']:
                if (camera_roi['represents'] not in match_roi):
                    match_roi[camera_roi['represents']] = []
                # здесь будут храниться детекции данного региона
                matched_detection = []
                for box, conf, class_id in detections[name]:
                    iou = get_iou(
                        list(map(float, camera_roi['xywh'])), list(map(float, box)))
                    # сохранили детекции с достаточной iou
                    if (iou > 1e-2):
                        matched_detection.append((box, name, iou))
                match_roi[camera_roi['represents']] += (matched_detection)

        # обходим все детекции и сравниваем каждую с каждой
        for roi, det in zip(match_roi, match_roi.values()):
            for i in range(len(det)):
                for j in range(i, len(det)):
                    if (i == j):
                        continue
                    a, a_cam, a_iou = det[i]
                    b, b_cam, b_iou = det[j]
                    # если это детекции с одной камеры - пропускаем
                    if (a_cam == b_cam):
                        continue
                    get_model_predict(a, b)
                    pass

        print(f'frame - {frame_index}')
        frame_index += 1

        # show_roi_detection(frame, list(map(float, box)), list(map(float, roi['xywh'])))
