import cv2
import numpy as np
from face_detector import detect_faces_dlib, detect_faces_dlib_cnn, detect_faces_haar, detect_faces_mtcnn
from face_descriptor import get_face_descriptor, get_face_descriptor_with_pca
from unique_visitors import UniqueVisitors

FRAME_WIDTH=1920
FRAME_HEIGHT=1080
FRAME_RATE=5
BRIGHTNESS=10
CONTRAST=11
SATURATION=12
HUE=13
GAIN=14
EXPOSURE=15



cap = cv2.VideoCapture(0)

if not (cap.isOpened()):
    print("Устройство не доступно!")

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

print("Width = ",cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print("Height = ",cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("Framerate = ",cap.get(cv2.CAP_PROP_FPS))
print("Format = ",cap.get(cv2.CAP_PROP_FORMAT))
cv2.namedWindow('Frame')
# Функции обратного вызова для трекбаров
def on_brightness_trackbar(val):
    cap.set(cv2.CAP_PROP_BRIGHTNESS, val / 255)

def on_contrast_trackbar(val):
    cap.set(cv2.CAP_PROP_CONTRAST, val / 255)

def on_saturation_trackbar(val):
    cap.set(cv2.CAP_PROP_SATURATION, val / 255)

def on_gain_trackbar(val):
    cap.set(cv2.CAP_PROP_GAIN, val / 255)

def on_hue_trackbar(val):
    cap.set(cv2.CAP_PROP_HUE, val)

def on_exposure_trackbar(val):
    cap.set(cv2.CAP_PROP_EXPOSURE, val / 255)

# Создаем трекбары
cv2.createTrackbar('Brightness', 'Frame', 0, 255, on_brightness_trackbar)
cv2.createTrackbar('Contrast', 'Frame', 0, 255, on_contrast_trackbar)
cv2.createTrackbar('Saturation', 'Frame', 0, 255, on_saturation_trackbar)
cv2.createTrackbar('Gain', 'Frame', 0, 255, on_gain_trackbar)
cv2.createTrackbar('Hue', 'Frame', 0, 179, on_hue_trackbar)
cv2.createTrackbar('Exposure', 'Frame', 0, 255, on_exposure_trackbar)


unique_visitors = UniqueVisitors()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    preview_scale = 0.5 
    preview_frame = cv2.resize(frame, (0, 0), fx=preview_scale, fy=preview_scale)

    rgb = cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)
    
    faces, shapes = detect_faces_mtcnn(rgb)
    
    for face, shape in zip(faces, shapes):

        descriptor = get_face_descriptor(preview_frame, shape)
        
        if unique_visitors.is_unique(descriptor):
            unique_visitors.add_face(descriptor)
        
        face_id = unique_visitors.get_id(descriptor)

        cv2.rectangle(preview_frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 1)
        cv2.putText(preview_frame, f"ID: {face_id}", (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    
    cv2.putText(preview_frame, f"Unique faces: {unique_visitors.count()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
    
    cv2.imshow('Frame', preview_frame)


    k = cv2.waitKey(1)
    if (k == 27):
        break
    elif k == ord('p'):
        print("******************************")
        print("Width = ",cap.get(FRAME_WIDTH))
        print("Height = ",cap.get(FRAME_HEIGHT))
        print("Framerate = ",cap.get(FRAME_RATE))
        print("Brightness = ",cap.get(BRIGHTNESS))
        print("Contrast = ",cap.get(CONTRAST))
        print("Saturation = ",cap.get(SATURATION))
        print("Gain = ",cap.get(GAIN))
        print("Hue = ",cap.get(HUE))
        print("Exposure = ",cap.get(EXPOSURE))
        print("******************************")
    
cap.release()
cv2.destroyAllWindows()


