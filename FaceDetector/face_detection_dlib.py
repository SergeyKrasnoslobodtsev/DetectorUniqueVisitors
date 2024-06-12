import os
import sys
from typing import List

import cv2
import dlib

from FaceDetector.detector import FaceDetector
from utils.logger import LoggerFactory
from utils.helpers import draw_bounding_box, convert_to_rgb


class FaceDetectorDlib(FaceDetector):
    cnn_model_filename = "G:/GitHub/DetectorUniqueVisitors/models/mmod_human_face_detector.dat"

    def __init__(self, model_type: str = "hog"):
        try:
            LoggerFactory.configure()

            self.logger = LoggerFactory.get_logger(FaceDetectorDlib.__name__)
            # load the model
            if model_type == "hog":
                self.face_detector = dlib.get_frontal_face_detector()
            else:
                # MMOD model
                self.face_detector = dlib.cnn_face_detection_model_v1(FaceDetectorDlib.cnn_model_filename)
            self.model_type = model_type
            self.logger.info("dlib: {} face detector loaded...".format(self.model_type))
        except Exception as e:
            raise e

    def detect_faces(self, image, num_upscaling: int = 1) -> List[List[int]]:
        return [
            self.dlib_rectangle_to_list(bbox)
            for bbox in self.face_detector(image, num_upscaling)
        ]

    def dlib_rectangle_to_list(self, dlib_bbox) -> List[int]:

        if type(dlib_bbox) == dlib.mmod_rectangle:
            dlib_bbox = dlib_bbox.rect
        # Top left corner
        x1, y1 = dlib_bbox.tl_corner().x, dlib_bbox.tl_corner().y
        width, height = dlib_bbox.width(), dlib_bbox.height()
        # Bottom right point
        x2, y2 = x1 + width, y1 + height

        return [x1, y1, x2, y2]


if __name__ == "__main__":
    # Sample Usage
    # ob = FaceDetectorDlib(model_type="hog")
    # img = cv2.imread("G:/GitHub/DetectorUniqueVisitors/test/1.jpg")
    # print(img.shape)
    # bbox = ob.detect_faces(convert_to_rgb(img))
    # print(bbox)
    # for box in bbox:
    #     draw_bounding_box(img, box)
    pass
