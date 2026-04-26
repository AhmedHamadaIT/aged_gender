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

    # Full-frame detection for task pipeline (associates faces to persons via IoU)
    pairs = engine.detect_and_embed_full_frame(bgr_frame, person_detections)
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

    Face pipeline (handled internally by InsightFace app.get()):
        1. RetinaFace detection → bounding boxes + 5-point landmarks
        2. Similarity-transform alignment using 5-point landmarks
        3. ArcFace embedding extraction on the aligned 112×112 face
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

        # Only load detection + recognition. The recognition module does its own
        # 5-point landmark alignment internally — no need for the heavy
        # landmark_2d_106 model (saves ~100MB GPU memory on Jetson).
        self._app = FaceAnalysis(
            name=model_name,
            allowed_modules=["detection", "recognition"],
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self._app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))
        self._face_counter = 0

        loaded_models = list(self._app.models.keys()) if isinstance(self._app.models, dict) else [getattr(m, "taskname", str(m)) for m in self._app.models]
        print(f"[FaceEngine] Ready — models: {loaded_models}")

    # ── Full-frame detection for task pipeline ─────────────────────────────

    def detect_and_embed_full_frame(
        self,
        frame: np.ndarray,
        person_detections: list,
        det_thresh: float = 0.5,
    ) -> List[Tuple["FaceDetection", object]]:
        """
        Run face detection on the **full frame** and associate each detected
        face with a tracked person via bounding box IoU overlap.

        This replaces the old approach of cropping each YOLO person bbox and
        running InsightFace per-crop. Benefits:
            - One inference pass instead of N (one per person)
            - Faces detected at native resolution with full context
            - InsightFace alignment works on the original image (not a crop)

        Args:
            frame:             BGR frame (full resolution).
            person_detections: List of Detection objects from FrameBus
                               (each has .x1/.y1/.x2/.y2 and .track_id).
            det_thresh:        Minimum face detection confidence.

        Returns:
            List of (FaceDetection, person_det) pairs.
            Faces that don't overlap any person bbox are skipped.
        """
        faces = self._app.get(frame)
        results = []

        h, w = frame.shape[:2]

        for face in faces:
            if float(face.det_score) < det_thresh:
                continue

            self._face_counter += 1
            fx1, fy1, fx2, fy2 = map(int, face.bbox)
            fx1, fy1 = max(0, fx1), max(0, fy1)
            fx2, fy2 = min(w, fx2), min(h, fy2)

            if fx2 <= fx1 or fy2 <= fy1:
                continue

            # Find the person bbox with the highest IoU overlap
            best_person = None
            best_iou    = 0.0

            for det in person_detections:
                iou = self._compute_iou(
                    (fx1, fy1, fx2, fy2),
                    (det.x1, det.y1, det.x2, det.y2),
                )
                # Face bbox should be contained within person bbox.
                # Use a low IoU threshold since face is much smaller than person.
                if iou > best_iou:
                    best_iou    = iou
                    best_person = det

            # Also accept containment: face center inside person bbox
            if best_person is None:
                face_cx = (fx1 + fx2) // 2
                face_cy = (fy1 + fy2) // 2
                for det in person_detections:
                    if det.x1 <= face_cx <= det.x2 and det.y1 <= face_cy <= det.y2:
                        best_person = det
                        break

            if best_person is None:
                continue

            yaw, pitch = self._estimate_pose(face)
            quality    = self._estimate_quality(face, frame)

            face_det = FaceDetection(
                face_id    = self._face_counter,
                bbox       = (fx1, fy1, fx2, fy2),
                confidence = float(face.det_score),
                quality    = quality,
                yaw        = yaw,
                pitch      = pitch,
                embedding  = face.normed_embedding if hasattr(face, "normed_embedding") else None,
            )
            results.append((face_det, best_person))

        return results

    # ── Standalone detection (used by API endpoints) ───────────────────────

    def detect_and_embed(
        self,
        frame: np.ndarray,
        det_thresh: float = 0.5,
    ) -> List[FaceDetection]:
        """
        Detect all faces in *frame*, extract embeddings.

        Returns a list of FaceDetection with bbox in frame coordinates.
        """
        faces = self._app.get(frame)
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
        faces = self._app.get(image)
        if not faces:
            return None

        # Pick the largest face by area
        best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return best.normed_embedding if hasattr(best, "normed_embedding") else None

    # ── IoU computation ───────────────────────────────────────────────────

    @staticmethod
    def _compute_iou(
        box_a: Tuple[int, int, int, int],
        box_b: Tuple[int, int, int, int],
    ) -> float:
        """Compute intersection-over-union between two (x1,y1,x2,y2) boxes."""
        xa = max(box_a[0], box_b[0])
        ya = max(box_a[1], box_b[1])
        xb = min(box_a[2], box_b[2])
        yb = min(box_a[3], box_b[3])

        inter = max(0, xb - xa) * max(0, yb - ya)
        if inter == 0:
            return 0.0

        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union  = area_a + area_b - inter

        return inter / union if union > 0 else 0.0

    # ── Pose estimation ───────────────────────────────────────────────────

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

    # ── Quality estimation ────────────────────────────────────────────────

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
