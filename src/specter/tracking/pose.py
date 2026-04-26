from __future__ import annotations
import numpy as np
from specter.tracking.models import ensure

POSE_CONNECTIONS = [
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24),
    (23, 24),
    (23, 25), (25, 27), (27, 31),
    (24, 26), (26, 28), (28, 32),
    (15, 17), (15, 19), (17, 19),
    (16, 18), (16, 20), (18, 20),
]


class PoseTracker:
    def __init__(self) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

        model_path = ensure("pose_landmarker_full.task")
        for delegate in (BaseOptions.Delegate.GPU, BaseOptions.Delegate.CPU):
            try:
                options = PoseLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(model_path), delegate=delegate),
                    running_mode=RunningMode.IMAGE,
                    num_poses=1,
                    min_pose_detection_confidence=0.6,
                    min_pose_presence_confidence=0.6,
                    min_tracking_confidence=0.5,
                )
                self._detector = PoseLandmarker.create_from_options(options)
                print(f"[specter/pose] {delegate.name}")
                break
            except Exception:
                continue

    def process(self, rgb_frame: np.ndarray) -> list[tuple[float, float, float]] | None:
        import mediapipe as mp
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._detector.detect(image)
        if not result.pose_landmarks:
            return None
        return [(lm.x, lm.y, lm.z) for lm in result.pose_landmarks[0]]

    def close(self) -> None:
        self._detector.close()
