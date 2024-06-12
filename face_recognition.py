from typing import List, Tuple, Any

import dlib

import numpy as np

from numpy import ndarray
from sklearn.cluster import DBSCAN

from utils.logger import LoggerFactory
from utils.helpers import convert_to_dlib_rectangle
from utils.exeptions import FaceMissing


class FaceRecognition:
    def __init__(self):
        LoggerFactory.configure()
        self.logger = LoggerFactory.get_logger(FaceRecognition.__name__)
        self.known_faces = {}
        self.face_id = 0
        self.unique_faces_count = 0
        self.shape_predictor = dlib.shape_predictor('G:/GitHub/DetectorUniqueVisitors/models/face_landmarks_68.dat')
        self.face_rec_model = dlib.face_recognition_model_v1('G:/GitHub/DetectorUniqueVisitors/models/dlib_face_recognition_resnet_model_v1.dat')
        self.cluster_model = None
        self.face_encodings = []  # Список всех эмбеддингов лиц
        self.cluster_labels = []  # Метки кластеров для известных лиц

    def register_face(self, image, bbox: List[int]) -> Tuple[int, float]:
        face_encoding = self.get_facial_fingerprint(image, bbox)

        # Если нет известных лиц, создаем новый кластер
        if not self.known_faces:
            self.face_id += 1
            self.unique_faces_count += 1
            self.known_faces[self.face_id] = [face_encoding]
            self.face_encodings.append(face_encoding)
            self.cluster_labels.append(self.face_id)
            self.logger.info(f'Face {self.face_id} recognition done with probability 1.0 (new face).')
            return self.face_id, 1.0

        # Поиск ближайшего кластера
        closest_face_id, min_distance = self.find_closest_face(face_encoding)
        if closest_face_id is not None and min_distance < 0.5:
            self.known_faces[closest_face_id].append(face_encoding)
            self.face_encodings.append(face_encoding)
            self.cluster_labels.append(closest_face_id)
            self.logger.info(f'Face {closest_face_id} recognition done with probability {1 - min_distance:.2f}.')
            return closest_face_id, 1 - min_distance

        # Регистрация нового лица
        self.face_id += 1
        self.unique_faces_count += 1
        self.known_faces[self.face_id] = [face_encoding]
        self.face_encodings.append(face_encoding)
        self.cluster_labels.append(self.face_id)
        self.logger.info(f'Face {self.face_id} recognition done with probability 1.0 (new face).')
        return self.face_id, 1.0

    def get_facial_fingerprint(self, image, bbox: List[int] = None) -> ndarray:
        if bbox is None:
            raise FaceMissing

        bbox = convert_to_dlib_rectangle(bbox)
        shape = self.shape_predictor(image, bbox)
        face_encoding = self.get_face_encoding(image, shape)
        return face_encoding

    def get_face_encoding(self, image, shape) -> ndarray:
        encoding = np.array(self.face_rec_model.compute_face_descriptor(image, shape, 1))
        return encoding

    @staticmethod
    def euclidean_distance(vector1: ndarray, vector2: ndarray) -> float:
        return np.linalg.norm(vector1 - vector2)

    def update_clusters(self):
        if len(self.face_encodings) < 2:
            return

        self.cluster_model = DBSCAN(eps=0.6, min_samples=2).fit(self.face_encodings)
        self.cluster_labels = self.cluster_model.labels_

    def find_closest_face(self, face_encoding: ndarray) -> Tuple[int, float]:
        min_distance = float('inf')
        closest_face_id = None
        for face_id, encodings in self.known_faces.items():
            for encoding in encodings:
                distance = self.euclidean_distance(encoding, face_encoding)
                if distance < min_distance:
                    min_distance = distance
                    closest_face_id = face_id
        return closest_face_id, min_distance

