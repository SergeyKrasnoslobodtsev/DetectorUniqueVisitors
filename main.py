import cv2
from face_detector import detect_faces_dlib, detect_faces_dlib_cnn, detect_faces_haar, detect_faces_mtcnn
from face_descriptor import get_face_descriptor, get_face_descriptor_with_pca
from unique_visitors import UniqueVisitors

FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not (cap.isOpened()):
    print("Устройство не доступно!")

cap.set(cv2.CAP_PROP_SETTINGS, 1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

unique_visitors = UniqueVisitors()

counter = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    preview_scale = 0.5
    preview_frame = cv2.resize(frame, (0, 0), fx=preview_scale, fy=preview_scale)

    rgb = cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)

    faces, shapes = detect_faces_mtcnn(rgb)

    if counter % 2 == 0:

        for face, shape in zip(faces, shapes):

            descriptor = get_face_descriptor(preview_frame, shape)

            if unique_visitors.is_unique(descriptor):
                unique_visitors.add_face(descriptor)

            face_id = unique_visitors.get_id(descriptor)

            cv2.rectangle(preview_frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 1)
            cv2.putText(preview_frame, f"ID: {face_id}", (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.putText(preview_frame, f"Unique faces: {unique_visitors.count()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.imshow('Frame', preview_frame)

    if cv2.waitKey(1) != 27:
        continue
    break

cap.release()
cv2.destroyAllWindows()
