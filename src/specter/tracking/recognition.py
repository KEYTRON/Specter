from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np

_DB_PATH = Path(__file__).parent.parent.parent.parent / "data" / "faces.npz"
THRESHOLD = 0.4  # cosine similarity — above = same person


@dataclass
class FaceMatch:
    name: str
    score: float          # 0.0 – 1.0
    bbox: tuple[int, int, int, int]  # x1 y1 x2 y2 in pixels


class FaceRecognizer:
    def __init__(self) -> None:
        import onnxruntime as ort
        from insightface.app import FaceAnalysis
        available = ort.get_available_providers()
        if "TensorrtExecutionProvider" in available:
            providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        elif "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        print(f"[specter/recognition] providers: {providers}")
        self._app = FaceAnalysis(providers=providers)
        self._app.prepare(ctx_id=0, det_size=(320, 320))
        self._names: list[str] = []
        self._embeddings: np.ndarray = np.empty((0, 512), dtype=np.float32)
        self._load_db()

    # --- enroll ----------------------------------------------------------------

    def enroll(self, name: str, bgr_frame: np.ndarray) -> str:
        faces = self._app.get(bgr_frame)
        if not faces:
            return "Лицо не найдено"
        # use the largest detected face
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        emb = face.embedding / np.linalg.norm(face.embedding)
        self._names.append(name)
        self._embeddings = np.vstack([self._embeddings, emb[np.newaxis]])
        self._save_db()
        return f"Зарегистрировано: {name}"

    def delete(self, name: str) -> None:
        indices = [i for i, n in enumerate(self._names) if n != name]
        self._names = [self._names[i] for i in indices]
        self._embeddings = self._embeddings[indices] if indices else np.empty((0, 512), dtype=np.float32)
        self._save_db()

    def enrolled_names(self) -> list[str]:
        return list(dict.fromkeys(self._names))  # unique, order preserved

    # --- identify --------------------------------------------------------------

    def identify(self, bgr_frame: np.ndarray) -> list[FaceMatch]:
        faces = self._app.get(bgr_frame)
        if not faces or self._embeddings.shape[0] == 0:
            return []
        results = []
        for face in faces:
            emb = face.embedding / np.linalg.norm(face.embedding)
            sims = self._embeddings @ emb            # cosine sim for all enrolled
            best_idx = int(np.argmax(sims))
            best_score = float(sims[best_idx])
            name = self._names[best_idx] if best_score >= THRESHOLD else "Неизвестно"
            bbox = tuple(int(v) for v in face.bbox)  # type: ignore[arg-type]
            results.append(FaceMatch(name=name, score=best_score, bbox=bbox))
        return results

    # --- persistence -----------------------------------------------------------

    def _save_db(self) -> None:
        _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez(_DB_PATH, names=np.array(self._names), embeddings=self._embeddings)

    def _load_db(self) -> None:
        if not _DB_PATH.exists():
            return
        data = np.load(_DB_PATH, allow_pickle=True)
        self._names = list(data["names"])
        self._embeddings = data["embeddings"].astype(np.float32)
