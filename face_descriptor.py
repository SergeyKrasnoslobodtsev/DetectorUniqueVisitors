import dlib
import cv2
import numpy as np
from sklearn.decomposition import PCA

face_rec_model = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat')

def get_face_descriptor(img, shape):
    return np.array(face_rec_model.compute_face_descriptor(img, shape))


pca = PCA(n_components=50)

def apply_pca_to_face(img, face):
    """
    Применяет PCA к области лица и возвращает уменьшенное изображение лица.
    
    Args:
        img (numpy.ndarray): Изображение.
        face (dlib.rectangle): Область лица.
    
    Returns:
        numpy.ndarray: Уменьшенное изображение лица.
    """
    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    face_img = img[y:y+h, x:x+w]
    face_img_resized = cv2.resize(face_img, (100, 100)).flatten()  # Изменение размера и преобразование в вектор
    face_img_pca = pca.fit_transform([face_img_resized])  # Применение PCA
    return face_img_pca

def get_face_descriptor_with_pca(img, shape):
    """
    Вычисляет дескриптор лица с использованием уменьшенного изображения лица.
    
    Args:
        img (numpy.ndarray): Изображение.
        shape (dlib.full_object_detection): Координаты лица.
    
    Returns:
        numpy.ndarray: Дескриптор лица.
    """
    face_descriptor = np.array(face_rec_model.compute_face_descriptor(img, shape))
    return face_descriptor