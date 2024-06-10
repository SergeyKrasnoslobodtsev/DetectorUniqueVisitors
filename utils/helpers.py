from typing import List, Tuple

import cv2
import dlib


def convert_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def convert_to_dlib_rectangle(bbox):
    return dlib.rectangle(bbox[0], bbox[1], bbox[2], bbox[3])


def draw_bounding_box(image, bbox: List[int], color: Tuple = (0, 255, 0)):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    return image


def draw_annotation(image, name: str, bbox: List[int], color: Tuple = (0, 255, 0)):
    draw_bounding_box(image, bbox, color=color)
    x1, y1, x2, y2 = bbox

    # Draw the label with name below the face
    cv2.rectangle(image, (x1, y2 - 20), (x2, y2), color, cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image, name, (x1 + 6, y2 - 6), font, 0.6, (0, 0, 0), 2)