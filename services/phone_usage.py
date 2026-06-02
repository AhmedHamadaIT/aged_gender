"""
services/phone_usage.py
-----------------------
PHONE_USAGE task — mobile phone usage detection.

For every person in the frame that falls inside the configured detection zone,
the task runs PhoneService on their crop and checks if a phone is detected.
An event is emitted for each person found using a phone.

Task config shape (from POST /api/tasks):
{
    "taskId"        : int,
    "taskName"      : str,
    "algorithmType" : "PHONE_USAGE",
    "channelId"     : int,
    "enable"        : bool,
    "threshold"     : int,        # 0-100 — min phone detection confidence
    "areaPosition"  : str,        # JSON-encoded array of polygon zone definitions
    "detailConfig"  : {},
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

from utils import build_image, draw_evidence_scene, make_evidence_paths
from utils.task_payload import task_frame_bgr
from utils.geometry import point_in_polygon_dict as _point_in_polygon  # M-9

# ── Constants ─────────────────────────────────────────────────────────────────

_WEEKDAY_MAP = {
    "MONDAY": 0, "TUESDAY": 1, "WEDNESDAY": 2, "THURSDAY": 3,
    "FRIDAY": 4, "SATURDAY": 5, "SUNDAY": 6,
}


# ── Task ──────────────────────────────────────────────────────────────────────

class PhoneUsageTask:

    def __init__(self, task_config: dict):
        self.task_id    = task_config["taskId"]
        self.task_name  = task_config["taskName"]
        self.channel_id = task_config["channelId"]
        self.threshold  = task_config.get("threshold", 50) / 100.0
        self.enable     = task_config.get("enable", True)

        self.zones = self._parse_zones(task_config.get("areaPosition", "[]"))

        raw_days            = task_config.get("validWeekday", list(_WEEKDAY_MAP.keys()))
        self.valid_weekdays = {_WEEKDAY_MAP[d] for d in raw_days if d in _WEEKDAY_MAP}
        self.valid_start_ms = task_config.get("validStartTime", 0)
        self.valid_end_ms   = task_config.get("validEndTime",   86400000)

        from services.phone import PhoneService
        self._phone = PhoneService()

        self._capture_dir = os.getenv("CAPTURE_DIR", "/local/storage/captures")
        self._scene_dir   = os.getenv("SCENE_DIR",   "/local/storage/scenes")
        self._events_dir  = os.getenv("EVENTS_DIR",  "/local/storage/events")
        os.makedirs(self._capture_dir, exist_ok=True)
        os.makedirs(self._scene_dir,   exist_ok=True)
        os.makedirs(self._events_dir,  exist_ok=True)

        self._jsonl_path = os.path.join(self._events_dir, f"task_{self.task_id}.jsonl")
        from pathlib import Path
        from utils.jsonl_writer import JsonlWriter as _JW
        self._jsonl_writer = _JW(Path(self._jsonl_path))

        print(
            f"[PhoneUsage/{self.task_id}] Ready — "
            f"threshold={self.threshold}, zones={len(self.zones)}"
        )

    # ── Main entry point ──────────────────────────────────────────────────────

    def __call__(self, payload: dict) -> list:
        if not self.enable or not self._in_schedule():
            return []

        frame     = task_frame_bgr(payload)
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

            # Run phone inference on this person crop
            context = {
                "data": {
                    "frame"    : frame,
                    "detection": {"items": [det], "count": 1},
                    "use_case" : {},
                }
            }
            context = self._phone(context)
            phone_results = context["data"]["use_case"].get("phone", [])

            # Check if any phone was detected on this person
            if phone_results and phone_results[0].phone_detected:
                # Get the highest confidence phone detection
                best_item = max(phone_results[0].items, key=lambda x: x["confidence"])
                conf_pct  = int(best_item["confidence"] * 100)
                zone      = self.zones[0] if self.zones else None
                event     = self._build_event(
                    det,
                    conf_pct,
                    best_item,
                    zone,
                    payload["timestamp"],
                    str(payload.get("camera_id") or ""),
                )
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

    def _build_event(
        self,
        det,
        conf_pct: int,
        phone_item: dict,
        zone: Optional[dict],
        timestamp: str,
        camera_id: str,
    ) -> dict:
        now_ms   = int(time.time() * 1000)
        event_id = hashlib.md5(
            f"{self.task_id}_{det.track_id}_{now_ms}".encode()
        ).hexdigest()

        cam_key = str(camera_id or self.channel_id or "unknown")
        cap_rel, scene_rel = make_evidence_paths(cam_key, event_id)

        x1, y1, x2, y2 = det.bbox
        area_points     = zone.get("point", []) if zone else []

        return {
            "eventId"     : event_id,
            "eventType"   : "PHONE_USAGE",
            "timestamp"   : now_ms,
            "timestampUTC": datetime.fromtimestamp(
                now_ms / 1000, tz=timezone.utc
            ).isoformat().replace("+00:00", "Z"),
            "taskId"      : self.task_id,
            "taskName"    : self.task_name,
            "channelId"   : self.channel_id,
            "alert": {
                "type"       : "phone_usage",
                "description": "Mobile phone usage detected",
                "confidence" : conf_pct,
            },
            "person": {
                "trackingId" : str(det.track_id),
                "boundingBox": {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1},
                "areaPoints" : area_points,
            },
            "phone": {
                "boundingBox": {
                    "x"     : phone_item["x1"],
                    "y"     : phone_item["y1"],
                    "width" : phone_item["x2"] - phone_item["x1"],
                    "height": phone_item["y2"] - phone_item["y1"],
                },
                "confidence": conf_pct,
            },
            "evidence": {
                "captureImage": build_image(cap_rel, "capture"),
                "sceneImage"  : build_image(scene_rel, "scene"),
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
        rel_cap = event["evidence"]["captureImage"]["path"]
        rel_sce = event["evidence"]["sceneImage"]["path"]
        capture_path = os.path.join(self._capture_dir, *rel_cap.split("/"))
        scene_path   = os.path.join(self._scene_dir,   *rel_sce.split("/"))
        os.makedirs(os.path.dirname(capture_path), exist_ok=True)
        os.makedirs(os.path.dirname(scene_path),   exist_ok=True)
        if crop.size > 0:
            cv2.imwrite(capture_path, crop)

        zone_points = event.get("person", {}).get("areaPoints") or []
        phone_bb    = event.get("phone", {}).get("boundingBox")
        sec_bbox    = None
        if phone_bb:
            px, py = int(phone_bb["x"]), int(phone_bb["y"])
            sec_bbox = (px, py, px + int(phone_bb["width"]), py + int(phone_bb["height"]))
        scene_vis = draw_evidence_scene(
            frame,
            subject_bbox=det.bbox,
            label=f"phone_usage id{det.track_id}",
            secondary_bbox=sec_bbox,
            secondary_label="phone" if sec_bbox else "",
            zone_points=zone_points if zone_points else None,
        )
        cv2.imwrite(scene_path, scene_vis)

        self._jsonl_writer.append(event)

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
            print(f"[PhoneUsage] Failed to parse areaPosition: {e}")
            return []
