# VRM/3D renderer — placeholder for Phase 2
# Will use ModernGL + pygltflib for VRM model loading and skeletal animation

class AvatarRenderer:
    """Loads and renders a VRM model, driven by tracking data."""

    def __init__(self) -> None:
        self._model = None

    def load_vrm(self, path: str) -> None:
        raise NotImplementedError("VRM loading — Phase 2")

    def apply_tracking(self, face_landmarks, hand_landmarks) -> None:
        raise NotImplementedError
