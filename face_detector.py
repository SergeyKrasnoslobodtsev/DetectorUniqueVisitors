import dlib
import cv2
from mtcnn_cv2 import MTCNN
from helpers.converters import convert_and_trim_bb

predictor = dlib.shape_predictor('models/face_landmarks_68.dat')

def detect_faces_dlib(img):
    detector = dlib.get_frontal_face_detector()
    faces = detector(img)
    shapes = [predictor(img, face) for face in faces]
    
    return faces, shapes

cnn_model_path = 'models\mmod_human_face_detector.dat'

def detect_faces_dlib_cnn(img):
    detector = dlib.cnn_face_detection_model_v1(cnn_model_path)
    faces = detector(img)
    shapes = [predictor(img, face.rect) for face in faces]
    faces = [face.rect for face in faces]
    return faces, shapes

haar_cascade = cv2.CascadeClassifier('models\haarcascade_frontalface_default.xml')

def detect_faces_haar(img):
    detections = haar_cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    faces = [dlib.rectangle(int(x), int(y), int(x+w), int(y+h)) for (x, y, w, h) in detections]
    shapes = [predictor(img, face) for face in faces]
    
    return faces, shapes


mtcnn_detector = MTCNN()

def detect_faces_mtcnn(img):
    detections = mtcnn_detector.detect_faces(img)
    faces = [dlib.rectangle(det['box'][0], det['box'][1], det['box'][0] + det['box'][2], det['box'][1] + det['box'][3]) for det in detections]
    shapes = [predictor(img, face) for face in faces]

    return faces, shapes

