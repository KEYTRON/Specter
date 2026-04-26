# Specter

Native VTuber application with real-time face, hand and body tracking.

Built with PyQt6 + MediaPipe. Runs natively on Linux — no browser tab, no Electron wrapper.

## Features

- **Face tracking** — 478 landmarks via MediaPipe FaceLandmarker
- **Hand tracking** — full skeleton for both hands (21 joints each)
- **Body pose** — 33-point full body pose via PoseLandmarker
- **Native GUI** — PyQt6, OpenGL viewport
- **Linux native** — direct V4L2 camera access via OpenCV
- **Decoupled pipeline** — camera runs at full FPS, tracking drops frames if busy

## Roadmap

- [ ] Phase 2 — 3D avatar renderer (Rust/wgpu + VRM model support)
- [ ] Phase 3 — lip sync, facial expressions, motion blending
- [ ] Phase 4 — full body skeletal animation driven by pose tracking

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- V4L2-compatible webcam
- `v4l2-utils` (for camera enumeration)

## Setup

```bash
git clone https://github.com/KEYTRON/Specter.git
cd Specter
uv sync
uv run specter
```

MediaPipe model files (~30 MB total) are downloaded automatically on first run.

## Architecture

```
src/specter/
├── main.py               # entry point
├── ui/
│   ├── main_window.py    # PyQt6 main window
│   └── viewport.py       # QOpenGLWidget — landmark visualizer
├── tracking/
│   ├── camera.py         # V4L2 capture + async tracking pipeline
│   ├── face.py           # FaceLandmarker (MediaPipe Tasks)
│   ├── hands.py          # HandLandmarker
│   └── pose.py           # PoseLandmarker (full body)
└── avatar/
    └── renderer.py       # VRM renderer — Phase 2
```

The tracking pipeline decouples camera capture from inference: frames are captured at full speed and dropped if the tracker is busy, keeping the preview smooth.

## License

MIT
