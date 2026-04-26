import queue
import threading
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage

# Run pose every N frames, recognition every M frames
_POSE_EVERY = 2
_REC_EVERY = 3


class TrackingWorker(threading.Thread):
    def __init__(self, face_tracker, hand_tracker, pose_tracker, recognizer, on_result):
        super().__init__(daemon=True)
        self._face = face_tracker
        self._hands = hand_tracker
        self._pose = pose_tracker
        self._recognizer = recognizer
        self._on_result = on_result
        self._queue: queue.Queue = queue.Queue(maxsize=1)
        self._running = True
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._frame = 0
        # cached results for skipped frames
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
            run_pose = self._frame % _POSE_EVERY == 0
            run_rec = self._frame % _REC_EVERY == 0

            # Submit all active models in parallel
            futures = {
                "face": self._executor.submit(self._face.process, rgb),
                "hands": self._executor.submit(self._hands.process, rgb),
            }
            if run_pose:
                futures["pose"] = self._executor.submit(self._pose.process, rgb)
            if run_rec and self._recognizer:
                futures["matches"] = self._executor.submit(self._recognizer.identify, bgr)

            result = {k: f.result() for k, f in futures.items()}

            # Carry forward cached values for skipped frames
            if "pose" not in result:
                result["pose"] = self._last_pose
            else:
                self._last_pose = result["pose"]

            if "matches" not in result:
                result["matches"] = self._last_matches
            else:
                self._last_matches = result["matches"]

            self._on_result(result)

    def stop(self) -> None:
        self._running = False
        self._executor.shutdown(wait=False)


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
