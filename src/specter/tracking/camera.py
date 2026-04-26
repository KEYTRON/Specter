import queue
import threading
import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage


class TrackingWorker(threading.Thread):
    """Runs all trackers in a background thread, decoupled from camera capture."""

    def __init__(self, face_tracker, hand_tracker, pose_tracker, on_result):
        super().__init__(daemon=True)
        self._face = face_tracker
        self._hands = hand_tracker
        self._pose = pose_tracker
        self._on_result = on_result
        self._queue: queue.Queue = queue.Queue(maxsize=1)
        self._running = True

    def submit(self, rgb_frame: np.ndarray) -> None:
        try:
            self._queue.put_nowait(rgb_frame)
        except queue.Full:
            pass  # drop frame if tracker is busy

    def run(self) -> None:
        while self._running:
            try:
                frame = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            result = {
                "face": self._face.process(frame),
                "hands": self._hands.process(frame),
                "pose": self._pose.process(frame),
            }
            self._on_result(result)

    def stop(self) -> None:
        self._running = False


class CameraThread(QThread):
    frame_ready = pyqtSignal(QImage)
    tracking_ready = pyqtSignal(dict)

    def __init__(self, camera_index: str, face_tracker, hand_tracker, pose_tracker) -> None:
        super().__init__()
        self._camera_index = camera_index
        self._face_tracker = face_tracker
        self._hand_tracker = hand_tracker
        self._pose_tracker = pose_tracker
        self._running = False

    def _on_tracking(self, result: dict) -> None:
        self.tracking_ready.emit(result)

    def run(self) -> None:
        worker = TrackingWorker(
            self._face_tracker,
            self._hand_tracker,
            self._pose_tracker,
            self._on_tracking,
        )
        worker.start()

        cap = cv2.VideoCapture(self._camera_index, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
        self._running = True

        while self._running:
            ok, frame = cap.read()
            if not ok:
                break

            # Mirror camera preview
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            worker.submit(rgb)

            h, w, ch = frame.shape
            qt_image = QImage(frame.data, w, h, ch * w, QImage.Format.Format_BGR888)
            self.frame_ready.emit(qt_image.copy())

        worker.stop()
        cap.release()

    def stop(self) -> None:
        self._running = False
