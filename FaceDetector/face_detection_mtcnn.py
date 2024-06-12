import cv2

from typing import List
from mtcnn import MTCNN

from FaceDetector.detector import FaceDetector
from utils.logger import LoggerFactory, capture_output
from utils.helpers import convert_to_rgb, draw_bounding_box
from utils.exeptions import InvalidImage, is_valid_img


class FaceDetectorMTCNN(FaceDetector):

    def __init__(self, crop_forehead: bool = True, shrink_ratio: int = 0.1):
        try:
            LoggerFactory.configure()

            self.logger = LoggerFactory.get_logger(FaceDetectorMTCNN.__name__)
            self.face_detector = MTCNN()
            self.crop_forehead = crop_forehead
            self.shrink_ratio = shrink_ratio
            self.logger.info('Initialized...')
        except Exception as e:
            raise e

    def detect_faces(self, image=None, conf_threshold: float = 0.99) -> List[List[int]]:

        if not is_valid_img(image):
            raise InvalidImage

        # detections: list = self.face_detector.detect_faces(image)

        # Оборачиваем вызов face_detector.detect_faces декоратором capture_output
        @capture_output
        def wrapped_detect_faces(image):
            return self.face_detector.detect_faces(image)

        detections: list = wrapped_detect_faces(image)

        boxes = []

        for _, detection in enumerate(detections):
            conf = detection["confidence"]
            if conf >= conf_threshold:
                x, y, w, h = detection["box"]
                x1, y1, x2, y2 = x, y, x + w, y + h
                self.logger.info(F'Detected face {_} with confidence {conf}')
                if self.crop_forehead:
                    y1 = y1 + int(h * self.shrink_ratio)
                boxes.append([x1, y1, x2, y2])

        return boxes


if __name__ == "__main__":
    face_detector = FaceDetectorMTCNN(crop_forehead=False)
    image = cv2.imread('G:/GitHub/DetectorUniqueVisitors/test/1.jpg')
    # image = cv2.imread('G:/GitHub/DetectorUniqueVisitors/test/2.jpg')
    # image = cv2.imread('G:/GitHub/DetectorUniqueVisitors/test/3.jpg')
    # image = cv2.imread('G:/GitHub/DetectorUniqueVisitors/test/4.jpg')
    boxes = face_detector.detect_faces(convert_to_rgb(image))

    for box in boxes:
        draw_bounding_box(image, box)
    preview_scale = 0.5
    preview_image = cv2.resize(image, (0, 0), fx=preview_scale, fy=preview_scale)
    cv2.imshow('Face Detector', preview_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    pass
