"""Downloads MediaPipe .task model files on first run."""

import urllib.request
from pathlib import Path

_MODELS_DIR = Path(__file__).parent.parent.parent.parent / "models"

_URLS = {
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    ),
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    ),
}


def ensure(name: str) -> Path:
    path = _MODELS_DIR / name
    if not path.exists():
        print(f"[specter] Скачиваю модель {name}...")
        _MODELS_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_URLS[name], path)
        print(f"[specter] {name} готово")
    return path
