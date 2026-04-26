from __future__ import annotations
import numpy as np


class HandTracker:
    def __init__(self) -> None:
        import mediapipe as mp
        self._mp_hands = mp.solutions.hands
        self._tracker = self._mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process(self, rgb_frame: np.ndarray) -> list[list[tuple[float, float, float]]] | None:
        result = self._tracker.process(rgb_frame)
        if not result.multi_hand_landmarks:
            return None
        return [
            [(lm.x, lm.y, lm.z) for lm in hand.landmark]
            for hand in result.multi_hand_landmarks
        ]

    def close(self) -> None:
        self._tracker.close()
