"""
services/face/face_service.py
-------------------------------
Pipeline service for face recognition.

Follows the same ``__call__(context)`` pattern as DetectorService, AgeGenderService, etc.
Integrates with the existing pipeline by reading detection results and writing face
recognition results into the context.

Pipeline flow:
    1. Read person detections from upstream YOLO detector
    2. For each person bbox, run InsightFace face detection inside the crop
    3. Filter faces by quality, yaw, pitch thresholds (from task config)
    4. Extract embeddings and search FAISS store for matches
    5. Handle known persons (recognition) and strangers (surveillance)
    6. Log events in JSONL and save evidence images
    7. Draw face bounding boxes + identity labels on frame

Reads:
    context["data"]["frame"]               — full BGR frame
    context["data"]["detection"]["items"]   — List[Detection] from YOLO

Writes:
    context["data"]["use_case"]["face"]    — List[dict] face event results
"""

import os
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from logger.logger_config import Logger

log = Logger.get_logger(__name__)

# BGR colors for face drawing
_COLOR_KNOWN   = (0, 200, 0)     # green — recognised person
_COLOR_STRANGER = (0, 0, 220)    # red   — unknown person
_COLOR_REJECTED = (128, 128, 128) # gray  — rejected face


@dataclass
class FaceEventResult:
    """Result of processing one face through the pipeline."""
    event      : dict           # full event dict (matches output spec)
    face_bbox  : tuple          # (x1, y1, x2, y2) in frame coordinates
    label      : str            # display label for annotation
    is_stranger: bool
    is_rejected: bool = False

    def to_dict(self) -> dict:
        return self.event


class FaceService:
    """
    Pipeline service for face recognition — the main integration point.

    Owns FaceEngine, FaceStore, FaceTaskManager, and FaceEventLogger as
    sub-components.  All are initialised lazily on first __call__ to avoid
    heavy model loading at import time.
    """

    def __init__(self):
        self.save = os.getenv("SAVE_OUTPUT", "True").lower() in ("true", "1", "yes")

        # Lazy-loaded components — set on first call
        self._engine  = None
        self._store   = None
        self._tasks   = None
        self._events  = None
        self._ready   = False

        # Per-face tracking for failed recognition attempts (for failCount logic)
        # Key: face embedding hash → count of consecutive failed matches
        self._fail_tracker: Dict[str, int] = defaultdict(int)

        # Attendance deduplication: (person_id, task_id) → last_check_in_timestamp
        self._attendance_dedup: Dict[tuple, float] = {}
        self._dedup_window_sec = 1800  # 30 minutes

        log.info("[FACE_SERVICE] Initialised (lazy load — models loaded on first frame)")

    def _ensure_ready(self):
        """Lazy-initialise heavy components on first use."""
        if self._ready:
            return

        from .face_engine import FaceEngine
        from .face_store  import FaceStore
        from .face_task   import FaceTaskManager
        from .face_events import FaceEventLogger

        log.info("[FACE_SERVICE] Loading models and stores...")
        self._engine = FaceEngine()
        self._store  = FaceStore()
        self._tasks  = FaceTaskManager()
        self._events = FaceEventLogger()
        self._ready  = True
        log.info("[FACE_SERVICE] Ready")

    # ── Properties for API access ─────────────

    @property
    def engine(self):
        self._ensure_ready()
        return self._engine

    @property
    def store(self):
        self._ensure_ready()
        return self._store

    @property
    def tasks(self):
        self._ensure_ready()
        return self._tasks

    @property
    def events(self):
        self._ensure_ready()
        return self._events

    # ── Pipeline entry point ──────────────────

    def __call__(self, context: Dict[str, Any]) -> Dict[str, Any]:
        self._ensure_ready()

        frame      = context["data"]["frame"]
        detections = context["data"]["detection"].get("items", [])

        face_results: List[FaceEventResult] = []

        # Get active tasks
        active_tasks = self._tasks.active_tasks()
        if not active_tasks:
            context["data"]["use_case"]["face"] = []
            return context

        for det in detections:
            # Only process person detections
            if det.class_name != "person":
                continue

            # Crop person region
            h, w = frame.shape[:2]
            x1 = max(0, det.x1)
            y1 = max(0, det.y1)
            x2 = min(w, det.x2)
            y2 = min(h, det.y2)

            if x2 <= x1 or y2 <= y1:
                continue

            person_crop = frame[y1:y2, x1:x2]

            # Detect faces within person crop
            faces = self._engine.detect_and_embed(person_crop, det_thresh=0.5)

            for face_det in faces:
                # Offset face coordinates from crop space to frame space
                face_det.offset(x1, y1)

                # Process this face against each active task
                for task in active_tasks:
                    result = self._process_face(task, face_det, frame)
                    if result is not None:
                        face_results.append(result)

        context["data"]["use_case"]["face"] = [r.to_dict() for r in face_results]

        # Draw face annotations on frame
        if self.save:
            context["data"]["frame"] = self._draw(frame, face_results)

        return context

    # ── Face processing pipeline ──────────────

    def _process_face(
        self,
        task     : Any,   # FaceTaskConfig
        face_det : Any,   # FaceDetection
        frame    : np.ndarray,
    ) -> Optional[FaceEventResult]:
        """
        Process a single detected face against task configuration.

        Steps:
            1. Check face pixel size threshold
            2. Check quality threshold
            3. Check yaw / pitch thresholds
            4. Extract embedding and search libraries
            5. Handle match (known) or no-match (stranger)
            6. Log event
        """
        cfg = task.detailConfig

        # ── Step 1: Face size check ──
        if face_det.width < cfg.facePixelSize or face_det.height < cfg.facePixelSize:
            return None  # too small, silently skip (not a rejection event)

        # ── Step 2: Quality check ──
        if face_det.quality < cfg.facePixelSize:
            event = self._events.log_rejection_event(
                task_id       = task.taskId,
                task_name     = task.taskName,
                channel_id    = task.channelId,
                face_det      = face_det,
                reason        = "face_quality_too_low",
                detail_config = cfg,
            )
            return FaceEventResult(
                event       = event,
                face_bbox   = face_det.bbox,
                label       = "Rejected (quality)",
                is_stranger = False,
                is_rejected = True,
            )

        # ── Step 3: Yaw / Pitch check ──
        if abs(face_det.yaw) > cfg.yawThreshold:
            event = self._events.log_rejection_event(
                task_id       = task.taskId,
                task_name     = task.taskName,
                channel_id    = task.channelId,
                face_det      = face_det,
                reason        = "yaw_too_large",
                detail_config = cfg,
            )
            return FaceEventResult(
                event       = event,
                face_bbox   = face_det.bbox,
                label       = "Rejected (yaw)",
                is_stranger = False,
                is_rejected = True,
            )

        if abs(face_det.pitch) > cfg.pitchThreshold:
            event = self._events.log_rejection_event(
                task_id       = task.taskId,
                task_name     = task.taskName,
                channel_id    = task.channelId,
                face_det      = face_det,
                reason        = "pitch_too_large",
                detail_config = cfg,
            )
            return FaceEventResult(
                event       = event,
                face_bbox   = face_det.bbox,
                label       = "Rejected (pitch)",
                is_stranger = False,
                is_rejected = True,
            )

        # ── Step 4: Embedding search ──
        if face_det.embedding is None:
            return None

        matches = self._store.search(
            embedding = face_det.embedding,
            lib_ids   = task.libIds,
            top_k     = 1,
            threshold = float(task.threshold),
        )

        # ── Step 5a: Known person matched ──
        if matches:
            best = matches[0]
            event = self._events.log_recognition_event(
                task_id      = task.taskId,
                task_name    = task.taskName,
                channel_id   = task.channelId,
                person_id    = best.person_id,
                person_name  = best.person_name,
                lib_id       = best.lib_id,
                face_det     = face_det,
                score        = best.score,
                frame        = frame,
            )

            # Attendance — deduplicate check-ins
            if task.taskName.lower() in ("attendance", "checkin", "check-in"):
                dedup_key = (best.person_id, task.taskId)
                now = time.time()
                last_checkin = self._attendance_dedup.get(dedup_key, 0)
                if now - last_checkin > self._dedup_window_sec:
                    self._attendance_dedup[dedup_key] = now
                    self._events.log_attendance_event(
                        task_id     = task.taskId,
                        task_name   = task.taskName,
                        channel_id  = task.channelId,
                        person_id   = best.person_id,
                        person_name = best.person_name,
                        lib_id      = best.lib_id,
                        score       = best.score,
                        face_det    = face_det,
                        frame       = frame,
                    )

            # Reset fail counter for this face
            self._fail_tracker.pop(face_det.face_id, None)

            return FaceEventResult(
                event       = event,
                face_bbox   = face_det.bbox,
                label       = f"{best.person_name} ({best.score:.0f}%)",
                is_stranger = False,
            )

        # ── Step 5b: No match — potential stranger ──
        if task.enableStranger:
            # Track failed attempts
            fail_key = face_det.face_id
            self._fail_tracker[fail_key] = self._fail_tracker.get(fail_key, 0) + 1
            fail_count = self._fail_tracker[fail_key]

            if fail_count >= cfg.failCount:
                # Add to stranger store
                face_crop = None
                x1, y1, x2, y2 = face_det.bbox
                fh, fw = frame.shape[:2]
                cx1, cy1 = max(0, x1), max(0, y1)
                cx2, cy2 = min(fw, x2), min(fh, y2)
                if cx2 > cx1 and cy2 > cy1:
                    face_crop = frame[cy1:cy2, cx1:cx2]

                stranger_id = self._store.add_stranger(
                    embedding  = face_det.embedding,
                    face_image = face_crop,
                    metadata   = {"taskId": task.taskId, "channelId": task.channelId},
                )

                event = self._events.log_stranger_event(
                    task_id      = task.taskId,
                    task_name    = task.taskName,
                    channel_id   = task.channelId,
                    stranger_id  = stranger_id,
                    face_det     = face_det,
                    fail_count   = fail_count,
                    frame        = frame,
                )

                # Reset fail counter after logging
                self._fail_tracker.pop(fail_key, None)

                return FaceEventResult(
                    event       = event,
                    face_bbox   = face_det.bbox,
                    label       = f"Stranger #{stranger_id}",
                    is_stranger = True,
                )

        return None

    # ── Drawing ───────────────────────────────

    def _draw(self, frame: np.ndarray, results: List[FaceEventResult]) -> np.ndarray:
        """Draw face bounding boxes, identity labels, and confidence on the frame."""
        out = frame.copy()

        for r in results:
            x1, y1, x2, y2 = r.face_bbox

            if r.is_rejected:
                color = _COLOR_REJECTED
            elif r.is_stranger:
                color = _COLOR_STRANGER
            else:
                color = _COLOR_KNOWN

            # Face bounding box
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            # Label background
            label = r.label
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(out, (x1, y1 - th - 10), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                out, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA,
            )

        return out
