from PyQt6.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QComboBox, QPushButton, QSplitter
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from specter.tracking.camera import CameraThread
from specter.ui.viewport import AvatarViewport


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Specter")
        self.resize(1280, 720)

        self._camera_thread: CameraThread | None = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # Left: avatar viewport
        self.viewport = AvatarViewport()
        splitter.addWidget(self.viewport)

        # Right: camera preview + controls
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 4, 4, 4)

        self.camera_label = QLabel("Камера не запущена")
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setMinimumSize(320, 240)
        self.camera_label.setStyleSheet("background: #111; color: #555;")
        right_layout.addWidget(self.camera_label)

        cam_row = QHBoxLayout()
        cam_row.addWidget(QLabel("Камера:"))
        self.cam_selector = QComboBox()
        self._populate_cameras()
        cam_row.addWidget(self.cam_selector)
        right_layout.addLayout(cam_row)

        self.btn_camera = QPushButton("Запустить камеру")
        self.btn_camera.clicked.connect(self._toggle_camera)
        right_layout.addWidget(self.btn_camera)

        self.status_label = QLabel("Готово")
        right_layout.addWidget(self.status_label)
        right_layout.addStretch()

        splitter.addWidget(right)
        splitter.setSizes([820, 460])

    def _populate_cameras(self) -> None:
        import subprocess, re
        self.cam_selector.clear()
        try:
            out = subprocess.check_output(["v4l2-ctl", "--list-devices"], text=True, stderr=subprocess.DEVNULL)
        except Exception:
            out = ""

        current_name = ""
        for line in out.splitlines():
            stripped = line.strip()
            if not stripped.startswith("/dev/video"):
                current_name = stripped.rstrip(":")
            else:
                dev = stripped
                label = f"{current_name} ({dev})" if current_name else dev
                self.cam_selector.addItem(label, dev)

    def _toggle_camera(self) -> None:
        if self._camera_thread and self._camera_thread.isRunning():
            self._stop_camera()
        else:
            self._start_camera()

    def _start_camera(self) -> None:
        cam_device = self.cam_selector.currentData()
        if cam_device is None:
            self.status_label.setText("Камера не найдена")
            return

        from specter.tracking.face import FaceTracker
        from specter.tracking.hands import HandTracker

        self._camera_thread = CameraThread(
            camera_index=cam_device,
            face_tracker=FaceTracker(),
            hand_tracker=HandTracker(),
        )
        self._camera_thread.frame_ready.connect(self._on_frame)
        self._camera_thread.tracking_ready.connect(self.viewport.update_tracking)
        self._camera_thread.start()
        self.btn_camera.setText("Остановить камеру")
        self.status_label.setText("Трекинг активен")

    def _stop_camera(self) -> None:
        if self._camera_thread:
            self._camera_thread.stop()
            self._camera_thread.wait()
            self._camera_thread = None
        self.camera_label.setText("Камера не запущена")
        self.btn_camera.setText("Запустить камеру")
        self.status_label.setText("Остановлено")

    def _on_frame(self, image: QImage) -> None:
        pixmap = QPixmap.fromImage(image).scaled(
            self.camera_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.camera_label.setPixmap(pixmap)

    def closeEvent(self, event) -> None:
        self._stop_camera()
        super().closeEvent(event)
