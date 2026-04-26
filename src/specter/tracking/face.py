from __future__ import annotations
import numpy as np
from specter.tracking.models import ensure


class FaceTracker:
    def __init__(self) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

        model_path = ensure("face_landmarker.task")
        base = BaseOptions(model_asset_path=str(model_path), delegate=BaseOptions.Delegate.GPU)
        try:
            options = FaceLandmarkerOptions(
                base_options=base,
                running_mode=RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.7,
                min_face_presence_confidence=0.7,
                min_tracking_confidence=0.6,
            )
            self._detector = FaceLandmarker.create_from_options(options)
            print("[specter/face] GPU delegate OK")
            return
        except Exception:
            pass
        # fallback to CPU
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.7,
            min_face_presence_confidence=0.7,
            min_tracking_confidence=0.6,
        )
        self._detector = FaceLandmarker.create_from_options(options)
        print("[specter/face] CPU fallback")

    def process(self, rgb_frame: np.ndarray) -> list[tuple[float, float, float]] | None:
        import mediapipe as mp
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._detector.detect(image)
        if not result.face_landmarks:
            return None
        return [(lm.x, lm.y, lm.z) for lm in result.face_landmarks[0]]

    def close(self) -> None:
        self._detector.close()
