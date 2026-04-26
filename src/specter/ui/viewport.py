from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QColor, QFont
import numpy as np


class AvatarViewport(QOpenGLWidget):
    """OpenGL viewport for avatar rendering. Starts as a debug landmark visualizer."""

    def __init__(self) -> None:
        super().__init__()
        self._face_landmarks: list | None = None
        self._hand_landmarks: list | None = None
        self.setMinimumSize(640, 480)

    def update_tracking(self, data: dict) -> None:
        self._face_landmarks = data.get("face")
        self._hand_landmarks = data.get("hands")
        self.update()

    def paintGL(self) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(20, 20, 30))

        w, h = self.width(), self.height()

        if not self._face_landmarks and not self._hand_landmarks:
            painter.setPen(QColor(80, 80, 100))
            painter.setFont(QFont("Sans", 14))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Аватар\n(трекинг не активен)")
            painter.end()
            return

        # Face landmarks
        if self._face_landmarks:
            painter.setPen(QColor(100, 200, 255))
            for lm in self._face_landmarks:
                x = int(lm[0] * w)
                y = int(lm[1] * h)
                painter.drawPoint(x, y)

        # Hand landmarks
        if self._hand_landmarks:
            for hand in self._hand_landmarks:
                painter.setPen(QColor(255, 180, 80))
                for lm in hand:
                    x = int(lm[0] * w)
                    y = int(lm[1] * h)
                    painter.drawEllipse(x - 3, y - 3, 6, 6)

        painter.end()
