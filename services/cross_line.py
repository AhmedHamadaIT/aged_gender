"""
services/cross_line.py
----------------------
CrossLine task — detects when a person crosses a configured virtual line.

Receives per-frame payloads from FrameBus. Each Detection in the payload
already carries a track_id assigned by BoT-SORT in FrameBus, so this task
only needs to track per-person line-side state and fire events on crossing.

Each crossing event is:
  - Returned to the caller (pushed to SSE stream by task_worker)
  - Written as a JSONL record to local storage
  - Saved as evidence images (person crop + full scene)

Task config shape (from POST /api/tasks):
{
    "taskId"        : int,
    "taskName"      : str,
    "algorithmType" : "CROSS_LINE",
    "channelId"     : int,
    "enable"        : bool,
    "threshold"     : int,            # 0-100 — minimum detection confidence
    "areaPosition"  : str,            # JSON-encoded array of line definitions
    "detailConfig"  : {
        "enableAttrDetect": bool,     # run age/gender on crossing person
        "enableReid"      : bool      # reserved
    },
    "validWeekday"  : List[str],
    "validStartTime": int,            # ms from midnight
    "validEndTime"  : int
}

areaPosition element:
{
    "line_id"  : str,
    "line_name": str,
    "point"    : [{"x": int, "y": int}, {"x": int, "y": int}],
    "direction": int   # 0=bidirectional, 1=A→B, 2=B→A
}
"""

import os
import json
import hashlib
import time
from datetime import datetime, timezone
from typing import Optional, Dict, Tuple

import cv2

# ── Schedule helpers ──────────────────────────────────────────────────────────

_WEEKDAY_MAP = {
    "MONDAY": 0, "TUESDAY": 1, "WEDNESDAY": 2, "THURSDAY": 3,
    "FRIDAY": 4, "SATURDAY": 5, "SUNDAY": 6,
}

# ── Geometry ──────────────────────────────────────────────────────────────────

def _line_side(point: Tuple, p1: Tuple, p2: Tuple) -> int:
    """
    Sign of the 2D cross product (p2-p1) × (point-p1).
    Returns  1 → left of directed line p1→p2
            -1 → right
             0 → on the line
    """
    cross = (p2[0] - p1[0]) * (point[1] - p1[1]) - (p2[1] - p1[1]) * (point[0] - p1[0])
    if cross > 0: return  1
    if cross < 0: return -1
    return 0


# ── CrossLine task ────────────────────────────────────────────────────────────

class CrossLineTask:

    def __init__(self, task_config: dict):
        self.task_id    = task_config["taskId"]
        self.task_name  = task_config["taskName"]
        self.channel_id = task_config["channelId"]
        self.threshold  = task_config.get("threshold", 50) / 100.0
        self.enable     = task_config.get("enable", True)

        detail           = task_config.get("detailConfig", {})
        self.enable_attr = detail.get("enableAttrDetect", False)
        self.enable_reid = detail.get("enableReid", False)   # reserved

        self.lines = self._parse_lines(task_config.get("areaPosition", "[]"))

        raw_days            = task_config.get("validWeekday", list(_WEEKDAY_MAP.keys()))
        self.valid_weekdays = {_WEEKDAY_MAP[d] for d in raw_days if d in _WEEKDAY_MAP}
        self.valid_start_ms = task_config.get("validStartTime", 0)
        self.valid_end_ms   = task_config.get("validEndTime",   86400000)

        # Per-track line-side state: {track_id (int): {line_id (str): side (int)}}
        self._track_sides: Dict[int, Dict[str, int]] = {}

        # Age/Gender — loaded only when enableAttrDetect is true
        self._age_gender = None
        if self.enable_attr:
            from services.age_gender import AgeGenderService
            self._age_gender = AgeGenderService()

        # Local storage
        self._capture_dir = os.getenv("CAPTURE_DIR", "/local/storage/captures")
        self._scene_dir   = os.getenv("SCENE_DIR",   "/local/storage/scenes")
        self._events_dir  = os.getenv("EVENTS_DIR",  "/local/storage/events")
        os.makedirs(self._capture_dir, exist_ok=True)
        os.makedirs(self._scene_dir,   exist_ok=True)
        os.makedirs(self._events_dir,  exist_ok=True)

        self._jsonl_path = os.path.join(self._events_dir, f"task_{self.task_id}.jsonl")

        print(
            f"[CrossLine/{self.task_id}] Ready — "
            f"{len(self.lines)} line(s), attr={self.enable_attr}"
        )

    # ── Main entry point ──────────────────────────────────────────────────────

    def __call__(self, payload: dict) -> list:
        if not self.enable or not self.lines or not self._in_schedule():
            return []

        frame     = payload["frame"]       # np.ndarray BGR
        detection = payload["detection"]

        persons = [
            d for d in detection.get("items", [])
            if d.class_name == "person"
            and d.confidence >= self.threshold
            and d.track_id != -1            # skip detections with no track yet
        ]

        events          = []
        active_track_ids = set()

        for det in persons:
            track_id = det.track_id
            active_track_ids.add(track_id)

            if track_id not in self._track_sides:
                self._track_sides[track_id] = {}

            for line in self.lines:
                crossing_dir = self._check_crossing(track_id, det.center, line)
                if crossing_dir is None:
                    continue

                attrs = self._get_attributes(frame, det)
                event = self._build_event(det, line, crossing_dir, attrs, payload["timestamp"])
                self._persist(event, frame, det)
                events.append(event)

        # Purge side-state for tracks no longer in the frame
        self._track_sides = {k: v for k, v in self._track_sides.items() if k in active_track_ids}

        return events

    # ── Line crossing ─────────────────────────────────────────────────────────

    def _check_crossing(self, track_id: int, centroid: Tuple, line: dict) -> Optional[int]:
        p1  = (line["point"][0]["x"], line["point"][0]["y"])
        p2  = (line["point"][1]["x"], line["point"][1]["y"])
        lid = line["line_id"]

        new_side  = _line_side(centroid, p1, p2)
        if new_side == 0:
            return None

        prev_side = self._track_sides[track_id].get(lid)
        self._track_sides[track_id][lid] = new_side

        if prev_side is None or prev_side == new_side:
            return None

        crossing_dir  = 1 if prev_side > 0 else 2
        direction_cfg = line.get("direction", 0)

        if direction_cfg == 0:
            return crossing_dir
        if direction_cfg == crossing_dir:
            return crossing_dir
        return None

    # ── Attribute detection ───────────────────────────────────────────────────

    def _get_attributes(self, frame, det) -> dict:
        if not self.enable_attr or self._age_gender is None:
            return {"gender": "Unknown", "age": "Unknown"}

        context = {
            "data": {
                "frame"    : frame,
                "detection": {"items": [det], "count": 1},
                "use_case" : {},
            }
        }
        context = self._age_gender(context)
        results = context["data"]["use_case"].get("age_gender", [])
        if results:
            r = results[0]
            return {"gender": r.gender, "age": r.age_group}
        return {"gender": "Unknown", "age": "Unknown"}

    # ── Event construction ────────────────────────────────────────────────────

    def _build_event(self, det, line: dict, crossing_dir: int, attrs: dict, timestamp: str) -> dict:
        now_ms   = int(time.time() * 1000)
        event_id = hashlib.md5(
            f"{self.task_id}_{det.track_id}_{now_ms}".encode()
        ).hexdigest()

        date_str     = datetime.now().strftime("%Y/%m/%d")
        capture_path = os.path.join(self._capture_dir, date_str, f"{event_id}_crop.jpg")
        scene_path   = os.path.join(self._scene_dir,   date_str, f"{event_id}_scene.jpg")

        x1, y1, x2, y2 = det.bbox

        return {
            "eventId"     : event_id,
            "eventType"   : "CROSS_LINE",
            "timestamp"   : now_ms,
            "timestampUTC": datetime.fromtimestamp(
                now_ms / 1000, tz=timezone.utc
            ).isoformat().replace("+00:00", "Z"),
            "taskId"      : self.task_id,
            "taskName"    : self.task_name,
            "channelId"   : self.channel_id,
            "line": {
                "id"       : line["line_id"],
                "name"     : line.get("line_name", ""),
                "direction": crossing_dir,
            },
            "person": {
                "trackingId" : str(det.track_id),
                "reidFeature": [],
                "boundingBox": {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1},
                "attributes" : attrs,
                "confidence" : int(det.confidence * 100),
            },
            "evidence": {
                "captureImage": capture_path,
                "sceneImage"  : scene_path,
            },
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def _persist(self, event: dict, frame, det):
        x1, y1, x2, y2 = det.bbox
        h, w  = frame.shape[:2]
        PAD   = 10
        crop  = frame[
            max(0, y1 - PAD): min(h, y2 + PAD),
            max(0, x1 - PAD): min(w, x2 + PAD),
        ]
        capture_path = event["evidence"]["captureImage"]
        scene_path   = event["evidence"]["sceneImage"]
        os.makedirs(os.path.dirname(capture_path), exist_ok=True)
        os.makedirs(os.path.dirname(scene_path),   exist_ok=True)
        if crop.size > 0:
            cv2.imwrite(capture_path, crop)
        cv2.imwrite(scene_path, frame)

        with open(self._jsonl_path, "a") as f:
            f.write(json.dumps(event) + "\n")

    # ── Schedule ──────────────────────────────────────────────────────────────

    def _in_schedule(self) -> bool:
        now = datetime.now()
        if now.weekday() not in self.valid_weekdays:
            return False
        ms_now = (now.hour * 3600 + now.minute * 60 + now.second) * 1000
        return self.valid_start_ms <= ms_now <= self.valid_end_ms

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_lines(area_position: str) -> list:
        try:
            return json.loads(area_position) if area_position else []
        except Exception as e:
            print(f"[CrossLine] Failed to parse areaPosition: {e}")
            return []
