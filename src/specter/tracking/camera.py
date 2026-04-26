import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage


class CameraThread(QThread):
    frame_ready = pyqtSignal(QImage)
    tracking_ready = pyqtSignal(dict)

    def __init__(self, camera_index: int, face_tracker, hand_tracker) -> None:
        super().__init__()
        self._camera_index = camera_index
        self._face_tracker = face_tracker
        self._hand_tracker = hand_tracker
        self._running = False

    def run(self) -> None:
        cap = cv2.VideoCapture(self._camera_index, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
        self._running = True

        while self._running:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            face = self._face_tracker.process(rgb)
            hands = self._hand_tracker.process(rgb)
            self.tracking_ready.emit({"face": face, "hands": hands})

            # Annotate preview frame
            if face:
                h, w = frame.shape[:2]
                for lm in face:
                    cv2.circle(frame, (int(lm[0] * w), int(lm[1] * h)), 1, (100, 200, 255), -1)

            h, w, ch = frame.shape
            qt_image = QImage(frame.data, w, h, ch * w, QImage.Format.Format_BGR888)
            self.frame_ready.emit(qt_image.copy())

        cap.release()

    def stop(self) -> None:
        self._running = False
