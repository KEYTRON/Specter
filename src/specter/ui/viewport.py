from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QColor, QFont, QPen

# Hand bone connections (21 landmarks per hand)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (0, 9), (9, 10), (10, 11), (11, 12),     # middle
    (0, 13), (13, 14), (14, 15), (15, 16),   # ring
    (0, 17), (17, 18), (18, 19), (19, 20),   # pinky
    (5, 9), (9, 13), (13, 17),               # palm
]

# Body bone connections (33 landmarks)
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


class AvatarViewport(QOpenGLWidget):
    def __init__(self) -> None:
        super().__init__()
        self._face_landmarks: list | None = None
        self._hand_landmarks: list | None = None
        self._pose_landmarks: list | None = None
        self.setMinimumSize(640, 480)

    def update_tracking(self, data: dict) -> None:
        self._face_landmarks = data.get("face")
        self._hand_landmarks = data.get("hands")
        self._pose_landmarks = data.get("pose")
        self.update()

    def paintGL(self) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(15, 15, 25))

        w, h = self.width(), self.height()
        has_data = self._face_landmarks or self._hand_landmarks or self._pose_landmarks

        if not has_data:
            painter.setPen(QColor(70, 70, 90))
            painter.setFont(QFont("Sans", 14))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Аватар\n(трекинг не активен)")
            painter.end()
            return

        def px(lm):
            return w - int(lm[0] * w), int(lm[1] * h)

        # Body skeleton
        if self._pose_landmarks:
            lms = self._pose_landmarks
            pen = QPen(QColor(80, 200, 120), 3)
            painter.setPen(pen)
            for a, b in POSE_CONNECTIONS:
                if a < len(lms) and b < len(lms):
                    painter.drawLine(*px(lms[a]), *px(lms[b]))
            painter.setPen(QColor(120, 255, 160))
            for lm in lms:
                x, y = px(lm)
                painter.drawEllipse(x - 4, y - 4, 8, 8)

        # Face landmarks — draw only key points (not all 478)
        if self._face_landmarks:
            KEY_FACE = [
                1, 4, 5, 6,          # nose
                33, 133, 362, 263,   # eye corners
                61, 291,             # mouth corners
                10, 152,             # forehead, chin
                234, 454,            # cheeks
            ]
            painter.setPen(QColor(100, 180, 255))
            for i, lm in enumerate(self._face_landmarks):
                x, y = px(lm)
                if i in KEY_FACE:
                    painter.drawEllipse(x - 4, y - 4, 8, 8)
                else:
                    painter.drawPoint(x, y)

        # Hand skeleton
        if self._hand_landmarks:
            colors = [QColor(255, 160, 60), QColor(255, 80, 160)]
            for hand_idx, hand in enumerate(self._hand_landmarks):
                color = colors[hand_idx % 2]

                # Bones
                pen = QPen(color, 2)
                painter.setPen(pen)
                for a, b in HAND_CONNECTIONS:
                    if a < len(hand) and b < len(hand):
                        painter.drawLine(*px(hand[a]), *px(hand[b]))

                # Joints — larger at knuckles
                for i, lm in enumerate(hand):
                    x, y = px(lm)
                    r = 6 if i in (0, 5, 9, 13, 17) else 4
                    painter.setBrush(color)
                    painter.drawEllipse(x - r, y - r, r * 2, r * 2)
                painter.setBrush(Qt.BrushStyle.NoBrush)

        painter.end()
