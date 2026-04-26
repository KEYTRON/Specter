import queue
import threading
from concurrent.futures import ThreadPoolExecutor, Future
import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage

_POSE_EVERY = 2   # run pose every N frames
_REC_EVERY = 3    # run recognition every N frames


class TrackingWorker(threading.Thread):
    """
    MediaPipe models (face/hands/pose) run sequentially — they share C++ global
    state and are not thread-safe.  InsightFace (ONNX Runtime) is thread-safe
    and runs concurrently in a separate executor thread.
    """

    def __init__(self, face_tracker, hand_tracker, pose_tracker, recognizer, on_result):
        super().__init__(daemon=True)
        self._face = face_tracker
        self._hands = hand_tracker
        self._pose = pose_tracker
        self._recognizer = recognizer
        self._on_result = on_result
        self._queue: queue.Queue = queue.Queue(maxsize=1)
        self._running = True
        self._rec_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rec")
        self._rec_future: Future | None = None
        self._frame = 0
        self._last_pose = None
        self._last_matches: list = []

    def submit(self, rgb: np.ndarray, bgr: np.ndarray) -> None:
        try:
            self._queue.put_nowait((rgb, bgr))
        except queue.Full:
            pass

    def run(self) -> None:
        while self._running:
            try:
                rgb, bgr = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            self._frame += 1

            # MediaPipe: sequential, single-threaded
            face = self._face.process(rgb)
            hands = self._hands.process(rgb)

            if self._frame % _POSE_EVERY == 0:
                self._last_pose = self._pose.process(rgb)

            # InsightFace: fire-and-forget in GPU thread, collect previous result
            if self._recognizer and self._frame % _REC_EVERY == 0:
                if self._rec_future is None or self._rec_future.done():
                    bgr_copy = bgr.copy()
                    self._rec_future = self._rec_executor.submit(
                        self._recognizer.identify, bgr_copy
                    )

            if self._rec_future is not None and self._rec_future.done():
                try:
                    self._last_matches = self._rec_future.result()
                except Exception:
                    self._last_matches = []
                self._rec_future = None

            self._on_result({
                "face": face,
                "hands": hands,
                "pose": self._last_pose,
                "matches": self._last_matches,
            })

    def stop(self) -> None:
        self._running = False
        self._rec_executor.shutdown(wait=False)


class CameraThread(QThread):
    frame_ready = pyqtSignal(QImage)
    tracking_ready = pyqtSignal(dict)

    def __init__(self, camera_index: str, face_tracker, hand_tracker, pose_tracker, recognizer=None) -> None:
        super().__init__()
        self._camera_index = camera_index
        self._face_tracker = face_tracker
        self._hand_tracker = hand_tracker
        self._pose_tracker = pose_tracker
        self._recognizer = recognizer
        self._running = False
        self._last_bgr: np.ndarray | None = None

    def last_bgr_frame(self) -> np.ndarray | None:
        return self._last_bgr.copy() if self._last_bgr is not None else None

    def _on_tracking(self, result: dict) -> None:
        self.tracking_ready.emit(result)

    def run(self) -> None:
        worker = TrackingWorker(
            self._face_tracker,
            self._hand_tracker,
            self._pose_tracker,
            self._recognizer,
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

            frame = cv2.flip(frame, 1)
            self._last_bgr = frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            worker.submit(rgb, frame)

            h, w, ch = frame.shape
            qt_image = QImage(frame.data, w, h, ch * w, QImage.Format.Format_BGR888)
            self.frame_ready.emit(qt_image.copy())

        worker.stop()
        cap.release()

    def stop(self) -> None:
        self._running = False
