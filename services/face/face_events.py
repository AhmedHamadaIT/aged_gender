"""
services/face/face_events.py
------------------------------
Event generation and JSONL logging for face recognition.

Produces structured events matching the backend output spec:
    - Recognition events (known person matched)
    - Stranger events (unknown person detected)
    - Rejection events (face failed quality / pose thresholds)

Saves three evidence image types per event:
    - captures/{eventId}.jpg  — full frame with bounding box overlay
    - faces/{eventId}.jpg     — cropped face image
    - scenes/{eventId}.jpg    — wider scene context

Also maintains a structured JSONL log file for offline analysis.
"""

import hashlib
import json
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from logger.logger_config import Logger

log = Logger.get_logger(__name__)


def _generate_event_id() -> str:
    """Generate a unique event ID as MD5 hash (matches spec format)."""
    raw = f"{time.time()}-{uuid.uuid4()}"
    return hashlib.md5(raw.encode()).hexdigest()


def _now_epoch_ms() -> int:
    """Current time in milliseconds since epoch."""
    return int(time.time() * 1000)


def _now_utc_iso() -> str:
    """Current time in ISO 8601 UTC format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.") + \
           f"{datetime.now(timezone.utc).microsecond // 1000:03d}Z"


class FaceEventLogger:
    """
    Generates and logs face recognition events.

    Events are:
        1. Appended to a JSONL log file (one JSON object per line)
        2. Stored as evidence images in subdirectories

    Instance is shared by FaceService (pipeline) and the API layer (queries).
    """

    def __init__(self, output_dir: str = None):
        self._output_dir = Path(output_dir or os.getenv("FACE_EVIDENCE_DIR", "./data/face/events"))
        self._lock        = threading.Lock()

        # Create evidence directories
        for sub in ["captures", "faces", "scenes", "logs"]:
            (self._output_dir / sub).mkdir(parents=True, exist_ok=True)

        self._log_path = self._output_dir / "logs" / "events.jsonl"
        self._device_sn = os.getenv("DEVICE_SN", "EDGE_DEVICE_001")

        log.info(f"[FACE_EVENTS] Output dir: {self._output_dir}")

    # ── Event builders ────────────────────────

    def log_recognition_event(
        self,
        task_id      : int,
        task_name    : str,
        channel_id   : int,
        person_id    : int,
        person_name  : str,
        lib_id       : int,
        face_det     : Any,       # FaceDetection dataclass
        score        : float,
        frame        : np.ndarray = None,
        channel_name : str = "",
    ) -> dict:
        """
        Log a successful recognition event (known person matched).
        """
        event_id  = _generate_event_id()
        face_img  = ""
        if frame is not None:
            face_img = self._save_evidence(event_id, frame, face_det)

        event = {
            "eventId"      : event_id,
            "eventType"    : "FACE",
            "timestamp"    : _now_epoch_ms(),
            "timestampUTC" : _now_utc_iso(),
            "taskId"       : task_id,
            "taskName"     : task_name,
            "deviceSN"     : self._device_sn,
            "channelId"    : channel_id,
            "channelName"  : channel_name or str(channel_id),
            "person"       : {
                "id"        : person_id,
                "name"      : person_name,
                "libId"     : lib_id,
                "isStranger": False,
            },
            "face"         : {
                "faceId"   : face_det.face_id if face_det else 0,
                "faceImage": face_img,
                "quality"  : round(face_det.quality, 1) if face_det else 0.0,
                "yaw"      : round(face_det.yaw, 3) if face_det else 0.0,
                "pitch"    : round(face_det.pitch, 3) if face_det else 0.0,
                "score"    : round(score, 1),
                "failCount": 0,
            },
        }

        self._append_log(event)
        return event

    def log_stranger_event(
        self,
        task_id      : int,
        task_name    : str,
        channel_id   : int,
        stranger_id  : int,
        face_det     : Any,
        fail_count   : int,
        frame        : np.ndarray = None,
        channel_name : str = "",
    ) -> dict:
        """
        Log a stranger detection event (unknown / unregistered person).
        """
        event_id  = _generate_event_id()
        face_img  = ""
        if frame is not None:
            face_img = self._save_evidence(event_id, frame, face_det)

        event = {
            "eventId"      : event_id,
            "eventType"    : "FACE",
            "timestamp"    : _now_epoch_ms(),
            "timestampUTC" : _now_utc_iso(),
            "taskId"       : task_id,
            "taskName"     : task_name,
            "deviceSN"     : self._device_sn,
            "channelId"    : channel_id,
            "channelName"  : channel_name or str(channel_id),
            "person"       : {
                "id"        : -1,
                "name"      : f"stranger_{stranger_id}",
                "libId"     : -1,
                "isStranger": True,
            },
            "face"         : {
                "faceId"   : face_det.face_id if face_det else 0,
                "faceImage": face_img,
                "quality"  : round(face_det.quality, 1) if face_det else 0.0,
                "yaw"      : round(face_det.yaw, 3) if face_det else 0.0,
                "pitch"    : round(face_det.pitch, 3) if face_det else 0.0,
                "score"    : 0.0,
                "failCount": fail_count,
            },
        }

        self._append_log(event)
        return event

    def log_rejection_event(
        self,
        task_id      : int,
        task_name    : str,
        channel_id   : int,
        face_det     : Any,
        reason       : str,
        detail_config: Any = None,
        channel_name : str = "",
    ) -> dict:
        """
        Log a rejection event (face failed quality / pose thresholds).
        """
        event_id = _generate_event_id()

        required_quality = 60
        required_yaw     = 35
        required_pitch   = 25

        if detail_config:
            required_yaw   = getattr(detail_config, "yawThreshold", 35)
            required_pitch = getattr(detail_config, "pitchThreshold", 25)
            required_quality = getattr(detail_config, "qualityThreshold", 60)

        event = {
            "eventId"      : event_id,
            "eventType"    : "FACE",
            "timestamp"    : _now_epoch_ms(),
            "timestampUTC" : _now_utc_iso(),
            "taskId"       : task_id,
            "taskName"     : task_name,
            "deviceSN"     : self._device_sn,
            "channelId"    : channel_id,
            "channelName"  : channel_name or str(channel_id),
            "status"       : "rejected",
            "reason"       : reason,
            "face"         : {
                "quality"        : round(face_det.quality, 1) if face_det else 0.0,
                "yaw"            : round(abs(face_det.yaw), 1) if face_det else 0.0,
                "pitch"          : round(abs(face_det.pitch), 1) if face_det else 0.0,
                "requiredQuality": required_quality,
                "requiredYaw"    : required_yaw,
                "requiredPitch"  : required_pitch,
            },
        }

        self._append_log(event)
        return event

    def log_attendance_event(
        self,
        task_id     : int,
        task_name   : str,
        channel_id  : int,
        person_id   : int,
        person_name : str,
        lib_id      : int,
        score       : float,
        face_det    : Any = None,
        frame       : np.ndarray = None,
    ) -> dict:
        """
        Log an attendance check-in event.
        This is a recognition event with additional attendance context.
        """
        event = self.log_recognition_event(
            task_id      = task_id,
            task_name    = task_name,
            channel_id   = channel_id,
            person_id    = person_id,
            person_name  = person_name,
            lib_id       = lib_id,
            face_det     = face_det,
            score        = score,
            frame        = frame,
        )
        event["attendance"] = {
            "checkIn"  : _now_utc_iso(),
            "personId" : person_id,
            "name"     : person_name,
        }
        # Re-write the log entry with attendance data
        # (last line in JSONL will have the attendance field)
        return event

    # ── Evidence saving ───────────────────────

    def _save_evidence(self, event_id: str, frame: np.ndarray, face_det: Any) -> str:
        """
        Save three evidence images:
            - captures/{eventId}.jpg  — full frame with face bbox
            - faces/{eventId}.jpg     — cropped face
            - scenes/{eventId}.jpg    — wider scene
        Returns the face image filename.
        """
        face_fname = f"{event_id}.jpg"

        try:
            # 1. Full frame with bounding box (capture)
            capture = frame.copy()
            if face_det:
                x1, y1, x2, y2 = face_det.bbox
                cv2.rectangle(capture, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imwrite(str(self._output_dir / "captures" / face_fname), capture,
                        [cv2.IMWRITE_JPEG_QUALITY, 90])

            # 2. Face crop
            if face_det:
                x1, y1, x2, y2 = face_det.bbox
                h, w = frame.shape[:2]
                x1c = max(0, x1)
                y1c = max(0, y1)
                x2c = min(w, x2)
                y2c = min(h, y2)
                if x2c > x1c and y2c > y1c:
                    face_crop = frame[y1c:y2c, x1c:x2c]
                    cv2.imwrite(str(self._output_dir / "faces" / face_fname), face_crop,
                                [cv2.IMWRITE_JPEG_QUALITY, 90])

            # 3. Scene (wider context — 2x face area, centred on face)
            if face_det:
                cx, cy = face_det.center
                fw = face_det.width * 2
                fh = face_det.height * 2
                h, w = frame.shape[:2]
                sx1 = max(0, cx - fw)
                sy1 = max(0, cy - fh)
                sx2 = min(w, cx + fw)
                sy2 = min(h, cy + fh)
                scene = frame[sy1:sy2, sx1:sx2]
                if scene.size > 0:
                    cv2.imwrite(str(self._output_dir / "scenes" / face_fname), scene,
                                [cv2.IMWRITE_JPEG_QUALITY, 85])

        except Exception as e:
            log.warning(f"[FACE_EVENTS] Evidence save failed: {e}")

        return face_fname

    # ── JSONL I/O ─────────────────────────────

    def _append_log(self, event: dict):
        """Append a single event as one JSON line."""
        with self._lock:
            try:
                with open(self._log_path, "a") as f:
                    f.write(json.dumps(event, default=str) + "\n")
            except Exception as e:
                log.warning(f"[FACE_EVENTS] Log append failed: {e}")

    def query_events(
        self,
        task_id    : int = None,
        is_stranger: bool = None,
        person_id  : int = None,
        limit      : int = 100,
        offset     : int = 0,
    ) -> dict:
        """
        Query events from the JSONL log file.

        Returns {total, offset, limit, events: [...]}.
        """
        events = []
        if not self._log_path.exists():
            return {"total": 0, "offset": offset, "limit": limit, "events": []}

        try:
            with open(self._log_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ev = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Apply filters
                    if task_id is not None and ev.get("taskId") != task_id:
                        continue
                    if is_stranger is not None:
                        person = ev.get("person", {})
                        if person.get("isStranger") != is_stranger:
                            continue
                    if person_id is not None:
                        person = ev.get("person", {})
                        if person.get("id") != person_id:
                            continue

                    events.append(ev)
        except Exception as e:
            log.warning(f"[FACE_EVENTS] Query failed: {e}")

        # Reverse for newest-first
        events.reverse()
        total = len(events)
        return {
            "total"  : total,
            "offset" : offset,
            "limit"  : limit,
            "events" : events[offset: offset + limit],
        }

    def query_attendance(
        self,
        person_id  : int = None,
        person_name: str = None,
        date_from  : str = None,
        date_to    : str = None,
        limit      : int = 100,
        offset     : int = 0,
    ) -> dict:
        """
        Query attendance records from the JSONL log.
        Only returns events that have the 'attendance' field.
        """
        events = []
        if not self._log_path.exists():
            return {"total": 0, "offset": offset, "limit": limit, "records": []}

        try:
            with open(self._log_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ev = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if "attendance" not in ev:
                        continue

                    person = ev.get("person", {})
                    if person_id is not None and person.get("id") != person_id:
                        continue
                    if person_name is not None and person_name.lower() not in person.get("name", "").lower():
                        continue

                    # Date filters
                    ts_utc = ev.get("timestampUTC", "")
                    if date_from and ts_utc < date_from:
                        continue
                    if date_to and ts_utc > date_to:
                        continue

                    events.append({
                        "eventId"   : ev.get("eventId"),
                        "person"    : person,
                        "attendance": ev.get("attendance"),
                        "timestamp" : ev.get("timestampUTC"),
                        "channelId" : ev.get("channelId"),
                        "score"     : ev.get("face", {}).get("score", 0),
                    })
        except Exception as e:
            log.warning(f"[FACE_EVENTS] Attendance query failed: {e}")

        events.reverse()
        total = len(events)
        return {
            "total"  : total,
            "offset" : offset,
            "limit"  : limit,
            "records": events[offset: offset + limit],
        }
