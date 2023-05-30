import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2


def show_detections(img: cv2.Mat, detections: pd.DataFrame):
    """Показывает все детекции на изображении"""
    _, ax = plt.subplots()
    for line in detections.values:
        x, y, w, h = line[2:6]
        rect = patches.Rectangle(
            (x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.imshow(img)
    plt.show()


def show_roi_detection(img: cv2.Mat, box, roi):
    _, ax = plt.subplots()
    x, y, w, h = box
    rect = patches.Rectangle(
        (x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    x, y, w, h = roi
    rect = patches.Rectangle(
        (x, y), w, h, linewidth=1, edgecolor='g', facecolor='none')
    ax.add_patch(rect)

    plt.imshow(img)
    plt.show()
