import time
from typing import List, Dict

import cv2
import numpy as np

from FaceDetector.face_detection_mtcnn import FaceDetectorMTCNN
from face_recognition import FaceRecognizer
from utils.helpers import convert_to_rgb
from utils.helpers import draw_annotation


class Recognizer:

    def __init__(
            self,
            model_loc: str = "models",
            face_detection_threshold: float = 0.8,
    ) -> None:

        self.face_recognizer = FaceRecognizer(
            model_loc=model_loc,
            face_detection_threshold=face_detection_threshold
        )

        self.face_detector = FaceDetectorMTCNN(crop_forehead=True, shrink_ratio=0.2)

    def recognize_face_video(
            self,
            video_path: str = None,
            detection_interval: int = 15,
            preview: bool = False,
            resize_scale: float = 0.5,
            verbose: bool = True
    ) -> None:
        if video_path is None:
            # If no video source is given, try
            # switching to webcam
            video_path = 0
        # elif not path_exists(video_path):
        #    raise FileNotFoundError

        cap, video_writer = None, None

        try:
            cap = cv2.VideoCapture(video_path)
            # To save the video file, get the opencv video writer
            # video_writer = get_video_writer(cap, output_path)
            frame_num = 1
            matches, name, match_dist = [], None, None

            # t1 = time.time()

            while True:
                status, frame = cap.read()
                if not status:
                    break
                try:
                    # Flip webcam feed so that it looks mirrored
                    if video_path == 0:
                        frame = cv2.flip(frame, 2)

                    if frame_num % detection_interval == 0:
                        # Scale down the image to increase model
                        # inference time.
                        smaller_frame = convert_to_rgb(
                            cv2.resize(frame, (0, 0), fx=resize_scale, fy=resize_scale)
                        )
                        # Detect faces
                        matches = self.face_recognizer.face_recognition(
                            image=smaller_frame, threshold=0.6, bboxes=None
                        )
                    if verbose:
                        self.annotate_facial_data(matches, frame, resize_scale)
                    if preview:
                        cv2.imshow("Preview", cv2.resize(frame, (680, 480)))

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                except Exception:
                    pass
                frame_num += 1

            # t2 = time.time()
            # logger.info("Time:{}".format((t2 - t1) / 60))
            # logger.info("Total frames: {}".format(frame_num))
            # logger.info("Time per frame: {}".format((t2 - t1) / frame_num))

        except Exception as exc:
            raise exc
        finally:
            cv2.destroyAllWindows()
            cap.release()

    @staticmethod
    def annotate_facial_data(
            matches: List[Dict], image, resize_scale: float
    ) -> None:
        for face_bbox, match, dist in matches:
            name = match["name"] if match is not None else "Unknown"
            # match_dist = '{:.2f}'.format(dist) if dist < 1000 else 'INF'
            # name = name + ', Dist: {}'.format(match_dist)
            # draw face labels
            draw_annotation(image, name, int(1 / resize_scale) * np.array(face_bbox))
