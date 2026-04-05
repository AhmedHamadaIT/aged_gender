"""
services/face/face_engine.py
-----------------------------
InsightFace wrapper — face detection, landmark extraction, and ArcFace embedding.

Runs face detection **within a provided image crop** (typically the person
bounding box from the upstream YOLO detector).  Returns face bounding boxes,
quality scores, head pose (yaw / pitch), and normalised 512-d embeddings.

Usage:
    engine = FaceEngine()
    faces  = engine.detect_and_embed(person_crop)
"""

import os
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from logger.logger_config import Logger

log = Logger.get_logger(__name__)


# ─────────────────────────────────────────────
# Face detection result
# ─────────────────────────────────────────────
@dataclass
class FaceDetection:
    """Single detected face with all metadata."""
    x1        : int
    y1        : int
    x2        : int
    y2        : int
    det_score : float              # detection confidence [0-1]
    landmarks : Optional[np.ndarray] = None   # (5, 2) — 5 facial landmarks
    quality   : float = 0.0        # face image quality [0-100]
    yaw       : float = 0.0        # head left-right rotation (degrees)
    pitch     : float = 0.0        # head up-down rotation (degrees)
    embedding : Optional[np.ndarray] = None   # 512-d normalised embedding

    # mutable ID assigned during processing
    face_id   : int = 0

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def area(self) -> int:
        return self.width * self.height

    def offset(self, dx: int, dy: int):
        """Translate bbox and landmarks by (dx, dy) — used to map from person
        crop coordinates back to full-frame coordinates."""
        self.x1 += dx
        self.y1 += dy
        self.x2 += dx
        self.y2 += dy
        if self.landmarks is not None:
            self.landmarks = self.landmarks.copy()
            self.landmarks[:, 0] += dx
            self.landmarks[:, 1] += dy

    def to_dict(self) -> dict:
        return {
            "faceId"   : self.face_id,
            "bbox"     : list(self.bbox),
            "quality"  : round(self.quality, 1),
            "yaw"      : round(self.yaw, 3),
            "pitch"    : round(self.pitch, 3),
            "score"    : round(self.det_score * 100, 1),
            "center"   : list(self.center),
            "width"    : self.width,
            "height"   : self.height,
        }


# ─────────────────────────────────────────────
# Head pose estimation from 5 landmarks
# ─────────────────────────────────────────────
def _estimate_pose_from_landmarks(landmarks: np.ndarray) -> Tuple[float, float]:
    """
    Rough yaw / pitch estimate from 5-point landmarks.

    Landmarks order (InsightFace convention):
        0 — left eye,  1 — right eye,
        2 — nose tip,
        3 — left mouth corner,  4 — right mouth corner

    Returns (yaw_degrees, pitch_degrees).
    """
    if landmarks is None or len(landmarks) < 5:
        return 0.0, 0.0

    left_eye    = landmarks[0]
    right_eye   = landmarks[1]
    nose        = landmarks[2]
    mouth_left  = landmarks[3]
    mouth_right = landmarks[4]

    # Yaw — how far the nose deviates from the midpoint of the eyes
    eye_center = (left_eye + right_eye) / 2.0
    eye_dist   = np.linalg.norm(right_eye - left_eye)
    if eye_dist < 1e-6:
        return 0.0, 0.0

    nose_offset = nose[0] - eye_center[0]
    yaw = math.degrees(math.atan2(nose_offset, eye_dist)) * 2.0

    # Pitch — nose vertical position relative to eye-mouth midpoint
    mouth_center = (mouth_left + mouth_right) / 2.0
    face_height  = mouth_center[1] - eye_center[1]
    if face_height < 1e-6:
        return yaw, 0.0

    nose_vert_offset = nose[1] - eye_center[1]
    expected_ratio   = 0.45  # nose is roughly 45% down from eyes to mouth
    actual_ratio     = nose_vert_offset / face_height
    pitch = math.degrees(math.atan2(actual_ratio - expected_ratio, 1.0)) * 60.0

    return float(yaw), float(pitch)


def _estimate_quality(face_img: np.ndarray, det_score: float) -> float:
    """
    Heuristic face quality score [0–100].

    Combines:
        - Detection confidence (40% weight)
        - Laplacian sharpness   (30% weight)
        - Face resolution       (30% weight)
    """
    # Detection confidence component
    det_component = det_score * 100.0 * 0.4

    # Sharpness (Laplacian variance)
    if face_img.size == 0:
        return det_component
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if len(face_img.shape) == 3 else face_img
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness = min(laplacian_var / 500.0, 1.0) * 100.0 * 0.3

    # Resolution component
    h, w = face_img.shape[:2]
    min_dim = min(h, w)
    resolution = min(min_dim / 112.0, 1.0) * 100.0 * 0.3

    return round(min(det_component + sharpness + resolution, 100.0), 1)


# ─────────────────────────────────────────────
# Face engine
# ─────────────────────────────────────────────
class FaceEngine:
    """
    InsightFace wrapper for detection + recognition.

    On init, loads the specified model pack (default ``buffalo_l``).
    Provides methods to detect faces and extract 512-d ArcFace embeddings.
    """

    _instance = None   # module-level singleton (heavy model — load once)

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        model_name : str = None,
        det_size   : int = None,
        ctx_id     : int = None,
    ):
        if self._initialized:
            return
        self._initialized = True

        model_name = model_name or os.getenv("FACE_MODEL", "buffalo_l")
        det_size   = det_size   or int(os.getenv("FACE_DET_SIZE", "640"))
        ctx_id     = ctx_id if ctx_id is not None else int(os.getenv("FACE_DEVICE", "0"))

        log.info(f"[FACE_ENGINE] Loading InsightFace model: {model_name}")
        log.info(f"[FACE_ENGINE] Detection size: {det_size}x{det_size}")
        log.info(f"[FACE_ENGINE] Device (ctx_id): {ctx_id}")

        try:
            import insightface
            from insightface.app import FaceAnalysis

            self.app = FaceAnalysis(
                name=model_name,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            self.app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))
            log.info(f"[FACE_ENGINE] Ready — models loaded: "
                     f"{[m.__class__.__name__ for m in self.app.models.values()]}")
        except Exception as e:
            log.error(f"[FACE_ENGINE] Failed to load: {e}")
            raise RuntimeError(f"FaceEngine init failed: {e}") from e

        self._face_id_counter = 0

    def _next_face_id(self) -> int:
        self._face_id_counter += 1
        return self._face_id_counter

    # ── Public API ────────────────────────────

    def detect_faces(
        self,
        image     : np.ndarray,
        det_thresh: float = 0.5,
        max_faces : int   = 50,
    ) -> List[FaceDetection]:
        """
        Detect faces in an image.

        Args:
            image:      BGR numpy array (full frame or person crop)
            det_thresh: minimum detection confidence
            max_faces:  maximum faces to return

        Returns:
            List of FaceDetection with bbox, landmarks, quality, yaw, pitch.
            Embeddings are NOT computed here — call get_embedding() separately if needed.
        """
        if image is None or image.size == 0:
            return []

        faces = self.app.get(image, max_num=max_faces)
        results = []

        for face in faces:
            if face.det_score < det_thresh:
                continue

            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

            # Clamp to image bounds
            h, w = image.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            landmarks = face.kps if hasattr(face, "kps") and face.kps is not None else None
            yaw, pitch = _estimate_pose_from_landmarks(landmarks)

            # Crop face for quality estimation
            face_crop = image[y1:y2, x1:x2]
            quality = _estimate_quality(face_crop, float(face.det_score))

            det = FaceDetection(
                x1        = x1,
                y1        = y1,
                x2        = x2,
                y2        = y2,
                det_score = float(face.det_score),
                landmarks = landmarks,
                quality   = quality,
                yaw       = yaw,
                pitch     = pitch,
                embedding = face.normed_embedding if hasattr(face, "normed_embedding") else None,
                face_id   = self._next_face_id(),
            )
            results.append(det)

        return results

    def get_embedding(self, image: np.ndarray, face_det: FaceDetection) -> Optional[np.ndarray]:
        """
        Extract embedding for a specific detected face.
        Returns 512-d normalised numpy array, or None on failure.

        Note: If detect_faces() already populated face_det.embedding (InsightFace
        does this by default when recognition model is loaded), this method
        simply returns the cached value.
        """
        if face_det.embedding is not None:
            return face_det.embedding

        # Fallback: re-run detection on face crop and extract embedding
        try:
            x1, y1, x2, y2 = face_det.bbox
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                return None

            faces = self.app.get(crop, max_num=1)
            if not faces:
                return None

            emb = faces[0].normed_embedding if hasattr(faces[0], "normed_embedding") else None
            face_det.embedding = emb
            return emb
        except Exception as e:
            log.warning(f"[FACE_ENGINE] Embedding extraction failed: {e}")
            return None

    def detect_and_embed(
        self,
        image     : np.ndarray,
        det_thresh: float = 0.5,
        max_faces : int   = 50,
    ) -> List[FaceDetection]:
        """
        Convenience: detect faces + extract embeddings in one pass.

        InsightFace's app.get() already extracts embeddings when the recognition
        model is loaded, so detect_faces() typically populates embeddings.
        This method explicitly verifies that.
        """
        detections = self.detect_faces(image, det_thresh, max_faces)

        for det in detections:
            if det.embedding is None:
                self.get_embedding(image, det)

        return detections

    def compute_embedding_from_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect the largest face in the image and return its embedding.
        Used for face registration (adding persons to library).

        Returns:
            512-d normalised numpy array, or None if no face found.
        """
        faces = self.detect_and_embed(image, det_thresh=0.3, max_faces=10)
        if not faces:
            return None

        # Return the largest face by area
        largest = max(faces, key=lambda f: f.area)
        return largest.embedding
