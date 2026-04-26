from __future__ import annotations
import numpy as np
from specter.tracking.models import ensure


class HandTracker:
    def __init__(self) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

        model_path = ensure("hand_landmarker.task")
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.IMAGE,
            num_hands=2,
        )
        self._detector = HandLandmarker.create_from_options(options)

    def process(self, rgb_frame: np.ndarray) -> list[list[tuple[float, float, float]]] | None:
        import mediapipe as mp
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._detector.detect(image)
        if not result.hand_landmarks:
            return None
        return [[(lm.x, lm.y, lm.z) for lm in hand] for hand in result.hand_landmarks]

    def close(self) -> None:
        self._detector.close()
