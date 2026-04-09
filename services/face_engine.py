"""
services/face_engine.py
-----------------------
InsightFace wrapper — face detection, embedding extraction, quality/pose.

Loaded once inside the task worker process (never at import time).
All configuration comes from environment variables with sensible defaults.

Usage:
    engine = FaceEngine()
    faces  = engine.detect_and_embed(bgr_frame)
    emb    = engine.embedding_from_image(bgr_image)
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class FaceDetection:
    face_id   : int
    bbox      : Tuple[int, int, int, int]   # (x1, y1, x2, y2)
    confidence: float
    quality   : float
    yaw       : float
    pitch     : float
    embedding : Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    def offset(self, dx: int, dy: int):
        """Shift bbox from crop-space to frame-space."""
        x1, y1, x2, y2 = self.bbox
        self.bbox = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)

    def to_dict(self) -> dict:
        return {
            "faceId"    : self.face_id,
            "bbox"      : list(self.bbox),
            "confidence": round(self.confidence, 3),
            "quality"   : round(self.quality, 1),
            "yaw"       : round(self.yaw, 3),
            "pitch"     : round(self.pitch, 3),
            "center"    : list(self.center),
            "width"     : self.width,
            "height"    : self.height,
        }


class FaceEngine:
    """
    InsightFace detection + ArcFace embedding engine.
    Initialised once per worker process — model loaded in __init__.
    """

    EMBEDDING_DIM = 512

    def __init__(self):
        import insightface
        from insightface.app import FaceAnalysis

        model_name = os.getenv("FACE_MODEL", "buffalo_l")
        det_size   = int(os.getenv("FACE_DET_SIZE", "640"))
        _device    = os.getenv("FACE_DEVICE", "0")
        ctx_id     = int(_device) if _device.isdigit() else -1

        print(f"[FaceEngine] Loading {model_name} (det_size={det_size}, ctx={ctx_id})")

        self._app = FaceAnalysis(
            name=model_name,
            allowed_modules=["detection", "recognition", "landmark_2d_106"],
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self._app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))
        self._face_counter = 0

        print(f"[FaceEngine] Ready — models: {[m.taskname for m in self._app.models]}")

    def detect_and_embed(
        self,
        frame: np.ndarray,
        det_thresh: float = 0.5,
    ) -> List[FaceDetection]:
        """
        Detect all faces in *frame*, extract embeddings.

        Returns a list of FaceDetection with bbox in frame coordinates.
        """
        faces = self._app.get(frame, det_thresh=det_thresh)
        results = []

        for face in faces:
            self._face_counter += 1
            x1, y1, x2, y2 = map(int, face.bbox)
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            yaw, pitch = self._estimate_pose(face)
            quality    = self._estimate_quality(face, frame)

            results.append(FaceDetection(
                face_id    = self._face_counter,
                bbox       = (x1, y1, x2, y2),
                confidence = float(face.det_score),
                quality    = quality,
                yaw        = yaw,
                pitch      = pitch,
                embedding  = face.normed_embedding if hasattr(face, "normed_embedding") else None,
            ))

        return results

    def embedding_from_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect the largest face in *image* and return its 512-d embedding.
        Returns None if no face is found.
        """
        faces = self._app.get(image, det_thresh=0.4)
        if not faces:
            return None

        # Pick the largest face by area
        best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return best.normed_embedding if hasattr(best, "normed_embedding") else None

    # ── Pose estimation ───────────────────────────────────────────────────────

    @staticmethod
    def _estimate_pose(face) -> Tuple[float, float]:
        """Estimate yaw/pitch from 2D landmarks (degrees)."""
        landmark = getattr(face, "landmark_2d_106", None)
        if landmark is None:
            landmark = getattr(face, "kps", None)

        if landmark is None:
            return (0.0, 0.0)

        try:
            if len(landmark) >= 106:
                left_eye  = landmark[33]
                right_eye = landmark[87]
                nose      = landmark[86]
            elif len(landmark) >= 5:
                left_eye  = landmark[0]
                right_eye = landmark[1]
                nose      = landmark[2]
            else:
                return (0.0, 0.0)

            eye_center = (left_eye + right_eye) / 2.0
            eye_dist   = np.linalg.norm(right_eye - left_eye)

            if eye_dist < 1e-6:
                return (0.0, 0.0)

            yaw   = float(np.degrees(np.arctan2(nose[0] - eye_center[0], eye_dist)))
            pitch = float(np.degrees(np.arctan2(eye_center[1] - nose[1], eye_dist)))
            return (yaw, pitch)
        except Exception:
            return (0.0, 0.0)

    # ── Quality estimation ────────────────────────────────────────────────────

    @staticmethod
    def _estimate_quality(face, frame: np.ndarray) -> float:
        """
        Heuristic quality score (0-100) based on:
        - Detection confidence
        - Face size relative to frame
        - Sharpness (Laplacian variance on face crop)
        """
        try:
            x1, y1, x2, y2 = map(int, face.bbox)
            h, w = frame.shape[:2]
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(w, x2), min(h, y2)

            if x2c <= x1c or y2c <= y1c:
                return 0.0

            crop = frame[y1c:y2c, x1c:x2c]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

            conf_score = float(face.det_score) * 40
            size_score = min((x2c - x1c) * (y2c - y1c) / (w * h) * 200, 30)
            sharp_score = min(sharpness / 10.0, 30)

            return min(conf_score + size_score + sharp_score, 100.0)
        except Exception:
            return float(face.det_score) * 100
