import cv2
import dlib

from FaceDetector.face_detection_mtcnn import FaceDetectorMTCNN
from FaceDetector.face_detection_dlib import FaceDetectorDlib
from face_recognition import FaceRecognition
from utils.helpers import draw_bounding_box, draw_annotation, convert_to_rgb

face_detector = FaceDetectorMTCNN(crop_forehead=True)
# face_detector = FaceDetectorDlib(model_type="mmod")
recognise = FaceRecognition()


# image = cv2.imread('./test/1.jpg')
# image = cv2.imread('./test/2.jpg')
# image = cv2.imread('./test/3.jpg')
image = cv2.imread('./test/4.jpg')
boxes = face_detector.detect_faces(convert_to_rgb(image))

for box in boxes:
    face_id, probability = recognise.register_face(image, box)
    draw_bounding_box(image, box)
    draw_annotation(image, f'ID - {face_id}', box)

cv2.putText(image, f'Unique faces: {recognise.unique_faces_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
            (0, 0, 0), 1)
cv2.imshow('Face Detector', image)

cv2.waitKey(0)

# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# cap.set(cv2.CAP_PROP_SETTINGS, 1)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# cap = cv2.VideoCapture('G:/GitHub/DetectorUniqueVisitors/test/street_people.mp4')
#
# counter = 0
#
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     preview_scale = 1
#     preview_frame = cv2.resize(frame, (0, 0), fx=preview_scale, fy=preview_scale)
#
#     if counter % 2 == 0:
#         boxes = face_detector.detect_faces(convert_to_rgb(preview_frame))
#
#         for box in boxes:
#             face_id, probability = recognise.register_face(preview_frame, box)
#             draw_bounding_box(preview_frame, box)
#             draw_annotation(preview_frame, f'ID: {face_id}', box)
#
#     cv2.putText(preview_frame, f'Unique faces: {recognise.unique_faces_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
#     cv2.imshow('Face Detector', preview_frame)
#
#     if cv2.waitKey(1) != 27:
#         continue
#     break
#
# cap.release()
cv2.destroyAllWindows()

