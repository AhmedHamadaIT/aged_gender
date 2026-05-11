"""
services/face_recognition.py
-----------------------------
FaceRecognitionTask — detects and recognises faces in each frame.

Follows the exact same contract as CrossLineTask:
    - __init__(task_config: dict) — called once in the worker process
    - __call__(payload: dict) -> list — called every frame, returns event dicts

Receives per-frame payloads from FrameBus. Each Detection in the payload
already carries a track_id assigned by BoT-SORT, which is used to avoid
re-processing the same face across consecutive frames.

Face detection pipeline:
    1. FrameBus sends the full frame + YOLO person detections (with track IDs)
    2. InsightFace runs ONCE on the full frame:
       - RetinaFace detection → face bounding boxes + 5-point landmarks
       - Similarity-transform alignment using 5-point landmarks
       - ArcFace 512-d embedding extraction on aligned 112×112 face
    3. Detected faces are associated to tracked persons via IoU / containment
    4. Each face is matched against registered libraries (FAISS cosine search)

Two operation modes:
    - Attendance  → match against registered face libraries, log check-ins
    - Surveillance → detect unknown persons, store in stranger index

Task config shape (from POST /api/tasks):
{
    "taskId"        : int,
    "taskName"      : str,
    "algorithmType" : "FACE",
    "channelId"     : int,
    "enable"        : bool,
    "threshold"     : int,            # 0-100 — minimum recognition score
    "libIds"        : str,            # "-1" = search all libraries
    "enableStranger": bool,
    "detailConfig"  : {
        "facePixelSize"  : int,       # min face size in pixels
        "yawThreshold"   : int,       # max allowed yaw (degrees)
        "pitchThreshold" : int,       # max allowed pitch (degrees)
        "failCount"      : int        # consecutive misses before stranger is logged
    },
    "validWeekday"  : List[str],
    "validStartTime": int,            # ms from midnight
    "validEndTime"  : int
}
"""

import os
import json
import hashlib
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional

import cv2
import numpy as np

from utils import build_image, draw_evidence_scene, make_evidence_paths

# ── Schedule helpers ──────────────────────────────────────────────────────────

_WEEKDAY_MAP = {
    "MONDAY": 0, "TUESDAY": 1, "WEDNESDAY": 2, "THURSDAY": 3,
    "FRIDAY": 4, "SATURDAY": 5, "SUNDAY": 6,
}


class FaceRecognitionTask:

    def __init__(self, task_config: dict):
        self.task_id    = task_config["taskId"]
        self.task_name  = task_config["taskName"]
        self.channel_id = task_config["channelId"]
        self.threshold  = task_config.get("threshold", 70)
        self.enable     = task_config.get("enable", True)
        self.lib_ids    = task_config.get("libIds", "-1")
        self.enable_stranger = task_config.get("enableStranger", True)

        detail = task_config.get("detailConfig", {})
        self.face_pixel_size  = detail.get("facePixelSize", 60)
        self.quality_threshold = detail.get("qualityThreshold", 60)
        self.yaw_threshold    = detail.get("yawThreshold", 35)
        self.pitch_threshold  = detail.get("pitchThreshold", 25)
        self.fail_count       = detail.get("failCount", 2)

        # Schedule
        raw_days            = task_config.get("validWeekday", list(_WEEKDAY_MAP.keys()))
        self.valid_weekdays = {_WEEKDAY_MAP[d] for d in raw_days if d in _WEEKDAY_MAP}
        self.valid_start_ms = task_config.get("validStartTime", 0)
        self.valid_end_ms   = task_config.get("validEndTime", 86400000)

        # ── Load engine and store inside worker process ──
        from services.face_engine import FaceEngine
        from services.face_store  import FaceStore

        self._engine = FaceEngine()
        self._store  = FaceStore()

        # Per-track state: {track_id: consecutive_fail_count}
        self._fail_tracker: Dict[int, int] = defaultdict(int)

        # Attendance dedup: (person_id, task_id) → last_checkin_epoch
        self._attendance_dedup: Dict[tuple, float] = {}
        self._dedup_window_sec = int(os.getenv("FACE_DEDUP_WINDOW", "1800"))  # 30 min

        # Evidence storage
        self._events_dir  = os.getenv("EVENTS_DIR", "/local/storage/events")
        self._capture_dir = os.getenv("CAPTURE_DIR", "/local/storage/captures")
        self._scene_dir   = os.getenv("SCENE_DIR", "/local/storage/scenes")
        os.makedirs(self._events_dir,  exist_ok=True)
        os.makedirs(self._capture_dir, exist_ok=True)
        os.makedirs(self._scene_dir,   exist_ok=True)

        self._jsonl_path = os.path.join(self._events_dir, f"task_{self.task_id}.jsonl")

        print(
            f"[Face/{self.task_id}] Ready — "
            f"threshold={self.threshold}%, libs={self.lib_ids}, "
            f"stranger={self.enable_stranger}"
        )

    # ── Main entry point ──────────────────────────────────────────────────────

    def __call__(self, payload: dict) -> list:
        """Called every frame by task_worker. Returns list of event dicts."""
        if not self.enable or not self._in_schedule():
            return []

        frame     = payload["frame"]
        detection = payload["detection"]
        timestamp = payload["timestamp"]
        events    = []

        persons = [
            d for d in detection.get("items", [])
            if d.class_name == "person"
            and d.track_id != -1
        ]

        active_track_ids = set()

        # ── Run InsightFace ONCE on the full frame ────────────────────────
        # Returns list of (FaceDetection, person_det) pairs, with each face
        # already associated to a tracked person via IoU / containment.
        face_person_pairs = self._engine.detect_and_embed_full_frame(
            frame, persons, det_thresh=0.5,
        )

        for face_det, person_det in face_person_pairs:
            active_track_ids.add(person_det.track_id)
            face_events = self._process_face(face_det, person_det, frame, timestamp)
            events.extend(face_events)

        # Also track persons with no face detected (for fail counting)
        for det in persons:
            active_track_ids.add(det.track_id)

        # Clean up fail tracker for tracks no longer in frame
        self._fail_tracker = {
            k: v for k, v in self._fail_tracker.items()
            if k in active_track_ids
        }

        return events

    # ── Face processing ───────────────────────────────────────────────────────

    def _process_face(self, face_det, person_det, frame, timestamp) -> list:
        """Process a single detected face. Returns 0 or 1 event dicts."""

        # Size check
        if face_det.width < self.face_pixel_size or face_det.height < self.face_pixel_size:
            return []

        # Quality check
        if face_det.quality < self.quality_threshold:
            return []

        # Yaw/Pitch check
        if abs(face_det.yaw) > self.yaw_threshold:
            return []
        if abs(face_det.pitch) > self.pitch_threshold:
            return []

        # Need embedding for matching
        if face_det.embedding is None:
            return []

        # Search libraries
        matches = self._store.search(
            embedding=face_det.embedding,
            lib_ids=self.lib_ids,
            top_k=1,
            threshold=float(self.threshold),
        )

        # ── Match found → recognition event ──
        if matches:
            best = matches[0]
            event = self._build_recognition_event(best, face_det, person_det, timestamp)
            self._persist(event, frame, face_det)

            # Reset fail counter
            self._fail_tracker.pop(person_det.track_id, None)

            # Attendance dedup
            if self.task_name.lower() in ("attendance", "checkin", "check-in"):
                dedup_key = (best.person_id, self.task_id)
                now = time.time()
                last = self._attendance_dedup.get(dedup_key, 0)
                if now - last > self._dedup_window_sec:
                    self._attendance_dedup[dedup_key] = now
                    event["attendance"] = {
                        "checkIn": datetime.fromtimestamp(
                            now, tz=timezone.utc
                        ).isoformat().replace("+00:00", "Z"),
                        "personId": best.person_id,
                        "name": best.person_name,
                    }

            return [event]

        # ── No match → potential stranger ──
        if self.enable_stranger:
            track_id = person_det.track_id
            self._fail_tracker[track_id] = self._fail_tracker.get(track_id, 0) + 1

            if self._fail_tracker[track_id] >= self.fail_count:
                face_crop = self._crop_face(frame, face_det)
                stranger_id = self._store.add_stranger(
                    embedding=face_det.embedding,
                    face_image=face_crop,
                    metadata={"taskId": self.task_id, "channelId": self.channel_id},
                )
                event = self._build_stranger_event(
                    stranger_id, face_det, person_det, timestamp)
                self._persist(event, frame, face_det)
                self._fail_tracker.pop(track_id, None)
                return [event]

        return []

    # ── Event builders ────────────────────────────────────────────────────────

    def _build_recognition_event(self, match, face_det, person_det, timestamp) -> dict:
        now_ms   = int(time.time() * 1000)
        event_id = hashlib.md5(
            f"{self.task_id}_{person_det.track_id}_{now_ms}".encode()
        ).hexdigest()

        cam_key = str(self.channel_id or "unknown")
        cap_rel, scene_rel = make_evidence_paths(cam_key, event_id)

        x1, y1, x2, y2 = face_det.bbox
        return {
            "eventId"     : event_id,
            "eventType"   : "FACE",
            "timestamp"   : now_ms,
            "timestampUTC": datetime.fromtimestamp(
                now_ms / 1000, tz=timezone.utc
            ).isoformat().replace("+00:00", "Z"),
            "taskId"      : self.task_id,
            "taskName"    : self.task_name,
            "channelId"   : self.channel_id,
            "deviceSN"    : os.getenv("DEVICE_SN", "EDGE_DEVICE_001"),
            "person": {
                "id"        : match.person_id,
                "name"      : match.person_name,
                "libId"     : match.lib_id,
                "isStranger": False,
                "trackingId": str(person_det.track_id),
                "boundingBox": {
                    "x": x1, "y": y1,
                    "width": x2 - x1, "height": y2 - y1,
                },
                "confidence": int(face_det.confidence * 100),
            },
            "face": {
                "faceId"   : face_det.face_id,
                "quality"  : round(face_det.quality, 1),
                "yaw"      : round(face_det.yaw, 3),
                "pitch"    : round(face_det.pitch, 3),
                "score"    : match.score,
                "failCount": 0,
            },
            "evidence": {
                "captureImage": build_image(cap_rel, "capture"),
                "sceneImage"  : build_image(scene_rel, "scene"),
            },
        }

    def _build_stranger_event(self, stranger_id, face_det, person_det, timestamp) -> dict:
        now_ms   = int(time.time() * 1000)
        event_id = hashlib.md5(
            f"{self.task_id}_stranger_{stranger_id}_{now_ms}".encode()
        ).hexdigest()

        cam_key = str(self.channel_id or "unknown")
        cap_rel, scene_rel = make_evidence_paths(cam_key, event_id)

        x1, y1, x2, y2 = face_det.bbox
        return {
            "eventId"     : event_id,
            "eventType"   : "FACE",
            "timestamp"   : now_ms,
            "timestampUTC": datetime.fromtimestamp(
                now_ms / 1000, tz=timezone.utc
            ).isoformat().replace("+00:00", "Z"),
            "taskId"      : self.task_id,
            "taskName"    : self.task_name,
            "channelId"   : self.channel_id,
            "deviceSN"    : os.getenv("DEVICE_SN", "EDGE_DEVICE_001"),
            "person": {
                "id"        : -1,
                "name"      : f"stranger_{stranger_id}",
                "libId"     : -1,
                "isStranger": True,
                "trackingId": str(person_det.track_id),
                "boundingBox": {
                    "x": x1, "y": y1,
                    "width": x2 - x1, "height": y2 - y1,
                },
                "confidence": int(face_det.confidence * 100),
            },
            "face": {
                "faceId"   : face_det.face_id,
                "quality"  : round(face_det.quality, 1),
                "yaw"      : round(face_det.yaw, 3),
                "pitch"    : round(face_det.pitch, 3),
                "score"    : 0.0,
                "failCount": self.fail_count,
            },
            "evidence": {
                "captureImage": build_image(cap_rel, "capture"),
                "sceneImage"  : build_image(scene_rel, "scene"),
            },
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def _persist(self, event: dict, frame: np.ndarray, face_det):
        """Save evidence images and append JSONL log."""
        rel_cap = event["evidence"]["captureImage"]["path"]
        rel_sce = event["evidence"]["sceneImage"]["path"]
        capture_path = os.path.join(self._capture_dir, *rel_cap.split("/"))
        scene_path   = os.path.join(self._scene_dir,   *rel_sce.split("/"))
        os.makedirs(os.path.dirname(capture_path), exist_ok=True)
        os.makedirs(os.path.dirname(scene_path),   exist_ok=True)

        crop = self._crop_face(frame, face_det)
        if crop is not None and crop.size > 0:
            cv2.imwrite(capture_path, crop)

        person_info = event.get("person", {})
        name  = person_info.get("name", "")
        label = f"{name} id{person_info.get('trackingId', '')}"
        scene_vis = draw_evidence_scene(frame, subject_bbox=face_det.bbox, label=label)
        cv2.imwrite(scene_path, scene_vis)

        with open(self._jsonl_path, "a") as f:
            f.write(json.dumps(event) + "\n")

    @staticmethod
    def _crop_face(frame: np.ndarray, face_det) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = face_det.bbox
        h, w = frame.shape[:2]
        PAD = 10
        y1c = max(0, y1 - PAD)
        y2c = min(h, y2 + PAD)
        x1c = max(0, x1 - PAD)
        x2c = min(w, x2 + PAD)
        if x2c <= x1c or y2c <= y1c:
            return None
        return frame[y1c:y2c, x1c:x2c]

    # ── Schedule ──────────────────────────────────────────────────────────────

    def _in_schedule(self) -> bool:
        now = datetime.now()
        if now.weekday() not in self.valid_weekdays:
            return False
        ms_now = (now.hour * 3600 + now.minute * 60 + now.second) * 1000
        return self.valid_start_ms <= ms_now <= self.valid_end_ms

    # ── Public accessors for API layer ────────────────────────────────────────

    @property
    def engine(self):
        return self._engine

    @property
    def store(self):
        return self._store
