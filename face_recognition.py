import os
from typing import List, Tuple, Any

import dlib
import numpy as np
from numpy import ndarray, dtype

from FaceDetector.face_detection_mtcnn import FaceDetectorMTCNN
from utils.helpers import convert_to_dlib_rectangle


class FaceRecognizer:
    landmarks_model_path = "shape_predictor_5_face_landmarks.dat"
    face_resnet_model_path = "dlib_face_recognition_resnet_model_v1.dat"

    def __init__(
            self,
            model_loc: str = "./models",
            face_detection_threshold: int = 0.99
    ):
        self.unique_faces = []
        self.unique_ids = []
        self.counter = 0

        landmarks_model_path = os.path.join(model_loc, FaceRecognizer.landmarks_model_path)
        face_resnet_model_path = os.path.join(model_loc, FaceRecognizer.face_resnet_model_path)

        self.face_detector = FaceDetectorMTCNN(crop_forehead=True, shrink_ratio=0.2)

        self.face_detection_threshold = face_detection_threshold

        self.landmarks_detector = dlib.shape_predictor(landmarks_model_path)
        self.face_recognition = dlib.face_recognition_model_v1(face_resnet_model_path)

    def register_face(self, image=None, name: str = None, bbox: List[int] = None):

        image = image.copy()
        face_encoding = None

        try:
            if bbox is None:
                boxes = self.face_detector.detect_faces(image=image)

                bbox = boxes[0]
            face_encoding = self.get_facial_fingerprint(image, bbox)

            # save the encoding with the name
            data = self.unique_faces
        except Exception as exc:
            raise exc
        return data

    def recognition_faces(self, image, threshold: float = 0.6, boxes: List[List[int]] = None):

        image = image.copy()

        if boxes is None:
            boxes = self.face_detector.detect_faces(image=image)

        unique_faces_data = self.unique_faces
        matches = []
        for bbox in boxes:
            face_encoding = self.get_facial_fingerprint(image, bbox)
            match, min_dist = None, 10000000

            for face_data in unique_faces_data:
                dist = self.euclidean_distance(face_encoding, face_data["encoding"])
                if dist <= threshold and dist < min_dist:
                    match = face_data
                    min_dist = dist

            matches.append((bbox, match, min_dist))
        return matches

    def get_facial_fingerprint(self, image, bbox: List[int] = None) -> ndarray[Any, dtype[Any]]:

        bbox = convert_to_dlib_rectangle(bbox)

        face_landmarks = self.landmarks_detector(image, bbox)

        face_encoding = self.get_face_encoding(image, face_landmarks)
        return face_encoding

    def get_face_encoding(self, image, face_landmarks: List):
        encoding = self.face_recognition.compute_face_descriptor(image, face_landmarks, 1)
        return np.array(encoding)

    @staticmethod
    def euclidean_distance(vector1: Tuple, vector2: Tuple):
        return np.linalg.norm(np.array(vector1) - np.array(vector2))
