from typing import List

from mtcnn import MTCNN

from FaceDetector.detector import FaceDetector


class FaceDetectorMTCNN(FaceDetector):

    def __init__(self, crop_forehead: bool = True, shrink_ratio: int = 0.1):
        try:
            self.face_detector = MTCNN()
            self.crop_forehead = crop_forehead
            self.shrink_ratio = shrink_ratio
            print("Загрузка детектора MTCNN...")
        except Exception as e:
            raise e

    def detect_faces(self, image, conf_threshold: float = 0.7) -> List[List[int]]:

        detections: list = self.face_detector.detect_faces(image)

        boxes = []
        
        for _, detection in enumerate(detections):
            conf = detection["confidence"]
            if conf >= conf_threshold:
                x, y, w, h = detection["box"]
                x1, y1, x2, y2 = x, y, x + w, y + h

                if self.crop_forehead:
                    y1 = y1 + int(h * self.shrink_ratio)
                boxes.append([x1, y1, x2, y2])

        return boxes
