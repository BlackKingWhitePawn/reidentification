import matplotlib.pyplot as plt
import cv2
import pandas as pd
from deep_sort_realtime.deepsort_tracker import DeepSort


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


def crop_from_frame(frame: cv2.Mat, box: tuple[float, float, float, float]):
    x, y, w, h = box
    return frame[int(y):int(y+h), int(x):int(x+w)]


def draw_detections(frame, detections):
    for box in detections:
        x, y, w, h = box[0]
        cv2.rectangle(
            frame, (int(x), int(y)),
            (int(x + w), int(y + h)), (255, 0, 0), 2
        )


def handle_boxes(boxes):
    boxes2 = boxes.copy()
    boxes2['left'] = boxes2['x1']
    boxes2['top'] = boxes2['y1']
    boxes2['w'] = boxes2['x2'] - boxes2['x1']
    boxes2['h'] = boxes2['y2'] - boxes2['y1']
    # boxes2 = boxes2.drop(labels=['confidence', 'class', 'x2', 'y2', 'x1', 'y1'], axis=1)
    return list(zip(zip(
        boxes2['left'].values,
        boxes2['top'].values,
        boxes2['w'].values,
        boxes2['h'].values
    ), boxes2['confidence'], boxes2['class']))


def run(camera_captures, calibration_data, detector, get_model_predict):
    """Запускает отслеживание на перечисленных камерах"""
    camera_data = list(map(lambda x: (x[0][0], x[1][0], x[1][1].split('.')[
                       0]), zip(camera_captures, calibration_data)))
    trackers = {}
    tracks = {}
    frame_index = 3480
    # храним текущие кадры
    current_frames = {}

    # словарь соответствия id межкамерных треков
    sync_track_ids = {

    }

    # создаем трекеры для каждой камеры
    for cap, calibration, name in camera_data:
        trackers[name] = DeepSort(max_age=30)
        # запсукаем окна
        # cv2.namedWindow(name)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    while True:
        # словарь текущих детекций для каждой камеры
        detections = {}
        # videos_end = True
        # для каждого кадра получаем список детекций и обновляем трекеры
        for cap, calibration, name in camera_data:
            _, frame = cap.read()
            # videos_end = videos_end and flag
            current_frames[name] = frame

            res = detector.predict(frame, imgsz=1280, classes=[0])
            bbs = handle_boxes(pd.DataFrame(
                res[0].boxes.data.cpu(),
                columns=['x1', 'y1', 'x2', 'y2', 'confidence', 'class'],
            ))
            draw_detections(frame, bbs)
            # cv2.imshow(name, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # сохраненине детекции
            detections[name] = bbs
            # обновление трека сортом
            tracks[name] = trackers[name].update_tracks(
                raw_detections=bbs, frame=frame)

        # if videos_end:
        #     return

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
                        track_id = list(
                            filter(lambda t: t.det_conf == conf, tracks[name]))[0].track_id
                        matched_detection.append((box, name, iou, track_id))
                match_roi[camera_roi['represents']] += (matched_detection)

        # обходим все детекции и сравниваем каждую с каждой
        for roi, det in zip(match_roi, match_roi.values()):
            for i in range(len(det)):
                for j in range(i, len(det)):
                    if (i == j):
                        continue
                    a, a_cam, a_iou, a_track_id = det[i]
                    b, b_cam, b_iou, b_track_id = det[j]
                    # если это детекции с одной камеры - пропускаем
                    if (a_cam == b_cam):
                        continue
                    img1 = crop_from_frame(current_frames[a_cam], a)
                    img2 = crop_from_frame(current_frames[b_cam], b)
                    cv2.imshow('test', cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
                    cv2.waitKey(0)
                    cv2.imshow('test', cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
                    cv2.waitKey(0)
                    res = get_model_predict(img1, img2)
                    # обнаружили совпадение
                    if (res):
                        # соединяем трекеры - переносим id с первой камеры на вторую
                        # for track in tracks[b_cam]:
                        if (not a_cam in sync_track_ids):
                            sync_track_ids[a_cam] = {}
                        if (not b_cam in sync_track_ids[a_cam]):
                            sync_track_ids[a_cam][b_cam] = {}
                        sync_track_ids[a_cam][b_cam][a_track_id] = b_track_id
                        # current_track.track_id = f'{a_cam}_{a_track_id}'
                        #     if track.is_confirmed():
                        #         track_id = track.track_id

        print(f'frame - {frame_index}')
        frame_index += 1

        # show_roi_detection(frame, list(map(float, box)), list(map(float, roi['xywh'])))
