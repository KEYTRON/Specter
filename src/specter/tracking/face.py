from __future__ import annotations
import numpy as np


class FaceTracker:
    def __init__(self) -> None:
        import mediapipe as mp
        self._mp_face = mp.solutions.face_mesh
        self._tracker = self._mp_face.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,  # enables iris + lips detail
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process(self, rgb_frame: np.ndarray) -> list[tuple[float, float, float]] | None:
        result = self._tracker.process(rgb_frame)
        if not result.multi_face_landmarks:
            return None
        lms = result.multi_face_landmarks[0].landmark
        return [(lm.x, lm.y, lm.z) for lm in lms]

    def close(self) -> None:
        self._tracker.close()
