from typing import List, Tuple

import cv2
import dlib


def convert_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def convert_to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def convert_to_dlib_rectangle(bbox):
    return dlib.rectangle(bbox[0], bbox[1], bbox[2], bbox[3])


def draw_bounding_box(image, bbox: List[int], color: Tuple = (239, 205, 171)):
    x1, y1, x2, y2 = bbox
    thickness = 1
    radius = 10

    # Draw top-left corner
    cv2.ellipse(image, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
    # Draw top-right corner
    cv2.ellipse(image, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
    # Draw bottom-right corner
    cv2.ellipse(image, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)
    # Draw bottom-left corner
    cv2.ellipse(image, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)

    # Draw top and bottom edges
    cv2.line(image, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
    cv2.line(image, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
    # Draw left and right edges
    cv2.line(image, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
    cv2.line(image, (x2, y1 + radius), (x2, y2 - radius), color, thickness)

    return image


def draw_annotation(image, name: str, bbox: List[int], color: Tuple = (239, 205, 171)):
    x1, y1, x2, y2 = bbox

    # Draw the label with name above the face
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    text_size = cv2.getTextSize(name, font, font_scale, font_thickness)[0]
    text_width, text_height = text_size[0], text_size[1]

    text_bg_height = text_height + 10
    cv2.rectangle(image, (x1, y1 - text_bg_height), (x1 + text_width, y1), color, cv2.FILLED)
    cv2.putText(image, name, (x1, y1 - 5), font, font_scale, (0, 0, 0), font_thickness)

    return draw_bounding_box(image, bbox, color=color)


def get_video_writer(video_stream, output_filename: str = "G:/GitHub/DetectorUniqueVisitors/test/output.avi"):
    # (Width, Height)
    dims = (int(video_stream.get(3)), int(video_stream.get(4)))
    video_writer = cv2.VideoWriter(
        output_filename,
        cv2.VideoWriter.fourcc('M','P', 'E','G'),
        30, (dims[0], dims[1]))

    if not video_writer.isOpened():
        print("Ошибка: Невозможно открыть файл для записи видео.")
        return None
    return video_writer
