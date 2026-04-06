"""
services/mask_hairnet_chef_hat.py
---------------------------------
MASK_HAIRNET_CHEF_HAT task — PPE compliance detection.

For every person in the frame that falls inside the configured detection zone,
the task runs PPEService on their crop and checks which PPE items are missing.
An event is emitted for each violation type that is listed in
detailConfig.alarmType.

Only violation (no_*) alarm types trigger events.
Positive detections (mask, chef_hat, hat) are never alerted on.

Alarm type → PPE model class mapping:
    "no_mask"     → requires "mask"     detected by PPEService
    "no_hat"      → requires "hairnet"  detected by PPEService
    "no_chef_hat" → requires "hairnet"  detected by PPEService
                    (chef_hat not a separate model class; hairnet is the proxy)

Task config shape (from POST /api/tasks):
{
    "taskId"        : int,
    "taskName"      : str,
    "algorithmType" : "MASK_HAIRNET_CHEF_HAT",
    "channelId"     : int,
    "enable"        : bool,
    "threshold"     : int,        # 0-100 — min PPE detection confidence
    "areaPosition"  : str,        # JSON-encoded array of polygon zone definitions
    "detailConfig"  : {
        "alarmType": ["no_mask", "no_chef_hat", "no_hat"]
    },
    "validWeekday"  : List[str],
    "validStartTime": int,
    "validEndTime"  : int
}
"""

import os
import json
import hashlib
import time
from datetime import datetime, timezone
from typing import List, Optional

import cv2
import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────

_WEEKDAY_MAP = {
    "MONDAY": 0, "TUESDAY": 1, "WEDNESDAY": 2, "THURSDAY": 3,
    "FRIDAY": 4, "SATURDAY": 5, "SUNDAY": 6,
}

# Maps each alarm type to the PPE model class name that must be detected.
# If the mapped class is absent from the crop inference → violation.
_ALARM_TO_PPE_CLASS = {
    "no_mask"    : "mask",
    "no_hat"     : "hairnet",
    "no_chef_hat": "hairnet",
}

_ALERT_DESCRIPTIONS = {
    "no_mask"    : "Face mask not detected",
    "no_chef_hat": "Chef hat not detected",
    "no_hat"     : "Hairnet not detected",
}


# ── Polygon helpers ───────────────────────────────────────────────────────────

def _point_in_polygon(point: tuple, polygon: list) -> bool:
    """Ray-casting algorithm — returns True if point is inside the polygon."""
    x, y   = point
    n      = len(polygon)
    inside = False
    px, py = polygon[-1]["x"], polygon[-1]["y"]
    for pt in polygon:
        cx, cy = pt["x"], pt["y"]
        if ((cy > y) != (py > y)) and (x < (px - cx) * (y - cy) / (py - cy + 1e-9) + cx):
            inside = not inside
        px, py = cx, cy
    return inside


# ── Task ──────────────────────────────────────────────────────────────────────

class MaskHairnetChefHatTask:

    def __init__(self, task_config: dict):
        self.task_id    = task_config["taskId"]
        self.task_name  = task_config["taskName"]
        self.channel_id = task_config["channelId"]
        self.threshold  = task_config.get("threshold", 50) / 100.0
        self.enable     = task_config.get("enable", True)

        detail          = task_config.get("detailConfig", {})
        raw_alarms      = detail.get("alarmType", list(_ALARM_TO_PPE_CLASS.keys()))
        # Only keep alarm types this task can handle
        self.alarm_types: List[str] = [a for a in raw_alarms if a in _ALARM_TO_PPE_CLASS]

        self.zones = self._parse_zones(task_config.get("areaPosition", "[]"))

        raw_days            = task_config.get("validWeekday", list(_WEEKDAY_MAP.keys()))
        self.valid_weekdays = {_WEEKDAY_MAP[d] for d in raw_days if d in _WEEKDAY_MAP}
        self.valid_start_ms = task_config.get("validStartTime", 0)
        self.valid_end_ms   = task_config.get("validEndTime",   86400000)

        from services.ppe import PPEService
        self._ppe = PPEService()

        self._capture_dir = os.getenv("CAPTURE_DIR", "/local/storage/captures")
        self._scene_dir   = os.getenv("SCENE_DIR",   "/local/storage/scenes")
        self._events_dir  = os.getenv("EVENTS_DIR",  "/local/storage/events")
        os.makedirs(self._capture_dir, exist_ok=True)
        os.makedirs(self._scene_dir,   exist_ok=True)
        os.makedirs(self._events_dir,  exist_ok=True)

        self._jsonl_path = os.path.join(self._events_dir, f"task_{self.task_id}.jsonl")

        print(
            f"[MaskHairnetChefHat/{self.task_id}] Ready — "
            f"alarmTypes={self.alarm_types}, zones={len(self.zones)}"
        )

    # ── Main entry point ──────────────────────────────────────────────────────

    def __call__(self, payload: dict) -> list:
        if not self.enable or not self.alarm_types or not self._in_schedule():
            return []

        frame     = payload["frame"]
        detection = payload["detection"]

        persons = [
            d for d in detection.get("items", [])
            if d.class_name == "person"
            and d.confidence >= self.threshold
            and d.track_id != -1
        ]

        events = []

        for det in persons:
            # Skip persons outside all configured zones (if zones are defined)
            if self.zones and not self._in_any_zone(det.center):
                continue

            # Run PPE inference on this person crop
            context = {
                "data": {
                    "frame"    : frame,
                    "detection": {"items": [det], "count": 1},
                    "use_case" : {},
                }
            }
            context = self._ppe(context)
            ppe_results = context["data"]["use_case"].get("ppe", [])

            # Collect detected PPE class names for this person
            detected_classes = set()
            ppe_conf_map     = {}   # class_name → highest confidence
            if ppe_results:
                for item in ppe_results[0].items:
                    name = item["class_name"]
                    conf = item["confidence"]
                    detected_classes.add(name)
                    ppe_conf_map[name] = max(ppe_conf_map.get(name, 0.0), conf)

            # Check each alarm type
            for alarm_type in self.alarm_types:
                required_class = _ALARM_TO_PPE_CLASS[alarm_type]
                if required_class not in detected_classes:
                    # Use threshold as fallback confidence when class absent
                    conf_pct = int(ppe_conf_map.get(required_class, self.threshold) * 100)
                    zone     = self.zones[0] if self.zones else None
                    event    = self._build_event(det, alarm_type, conf_pct, zone, payload["timestamp"])
                    self._persist(event, frame, det)
                    events.append(event)

        return events

    # ── Zone filtering ────────────────────────────────────────────────────────

    def _in_any_zone(self, centroid: tuple) -> bool:
        for zone in self.zones:
            pts = zone.get("point", [])
            if len(pts) >= 3 and _point_in_polygon(centroid, pts):
                return True
        return False

    # ── Event construction ────────────────────────────────────────────────────

    def _build_event(self, det, alarm_type: str, conf_pct: int, zone: Optional[dict], timestamp: str) -> dict:
        now_ms   = int(time.time() * 1000)
        event_id = hashlib.md5(
            f"{self.task_id}_{det.track_id}_{alarm_type}_{now_ms}".encode()
        ).hexdigest()

        date_str     = datetime.now().strftime("%Y/%m/%d")
        capture_path = os.path.join(self._capture_dir, date_str, f"{event_id}_crop.jpg")
        scene_path   = os.path.join(self._scene_dir,   date_str, f"{event_id}_scene.jpg")

        x1, y1, x2, y2 = det.bbox
        area_points     = zone.get("point", []) if zone else []

        return {
            "eventId"     : event_id,
            "eventType"   : "MASK_HAIRNET_CHEF_HAT",
            "timestamp"   : now_ms,
            "timestampUTC": datetime.fromtimestamp(
                now_ms / 1000, tz=timezone.utc
            ).isoformat().replace("+00:00", "Z"),
            "taskId"      : self.task_id,
            "taskName"    : self.task_name,
            "channelId"   : self.channel_id,
            "alert": {
                "type"       : alarm_type,
                "description": _ALERT_DESCRIPTIONS.get(alarm_type, alarm_type),
                "confidence" : conf_pct,
            },
            "person": {
                "trackingId" : str(det.track_id),
                "boundingBox": {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1},
                "areaPoints" : area_points,
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
    def _parse_zones(area_position: str) -> list:
        try:
            return json.loads(area_position) if area_position else []
        except Exception as e:
            print(f"[MaskHairnetChefHat] Failed to parse areaPosition: {e}")
            return []
