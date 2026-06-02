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
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from utils import build_image, draw_evidence_scene
from utils.task_payload import task_frame_bgr
from utils.geometry import point_in_polygon_dict as _point_in_polygon  # M-9

# ── Constants ─────────────────────────────────────────────────────────────────

ALGORITHM_TYPE = "MASK_HAIRNET_CHEF_HAT"

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


def _env_truthy(name: str) -> bool:
    v = os.getenv(name)
    if v is None:
        return False
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _channel_id_data(ch_raw: Any) -> Any:
    """Eyego ``data.channelId``: int when numeric, else string."""
    if ch_raw is None:
        return None
    s = str(ch_raw).strip()
    if not s:
        return None
    try:
        return int(s)
    except (TypeError, ValueError):
        return s


def build_ppe_person_structural(
    alarm_type: str,
    area_points: list,
    bbox: tuple,
    score: int,
) -> str:
    """
    Eyego ``personStructural`` for PPE violations.

    ``areaPoints`` is a JSON-encoded string (not a native array) per integration spec.
    """
    x1, y1, x2, y2 = bbox
    ps_obj = {
        "alarmType"    : alarm_type,
        "areaPoints"   : json.dumps(area_points, separators=(",", ":")),
        "objectHeight" : int(y2 - y1),
        "objectWidth"  : int(x2 - x1),
        "objectX"      : int(x1),
        "objectY"      : int(y1),
        "score"        : int(score),
        "smokingHeight": 0,
        "smokingWidth" : 0,
        "smokingX"     : 0,
        "smokingY"     : 0,
    }
    compact = _env_truthy("PPE_COMPACT_PERSON_STRUCTURAL")
    if compact or not _env_truthy("PPE_PRETTY_PERSON_STRUCTURAL"):
        return json.dumps(ps_obj, separators=(",", ":"), ensure_ascii=False)
    return json.dumps(ps_obj, indent=2, ensure_ascii=False)


def build_ppe_spec_data(
    *,
    alarm_type: str,
    area_points: list,
    bbox: tuple,
    score: int,
    task_id: int,
    task_name: str,
    channel_id: Any,
    channel_name: str = "",
    device_sn: str = "",
    record_ms: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Eyego ``data`` block for ``MASK_HAIRNET_CHEF_HAT``.

    URL pattern (integration): ``{base}/{captureId}{id}.jpg`` where ``id`` is a
    32-char hex correlation id (separate from the UUID embedded in ``captureId``).
    Bases: ``PPE_CLOUD_IMAGE_BASE``, then ``PPE_CAPTURE_URL_BASE`` /
    ``PPE_SCENE_URL_BASE``, with ``PPE_FORCE_LOCAL_URLS`` fallback.
    """
    ch_out = _channel_id_data(channel_id)
    ch_str = str(channel_id).strip() if channel_id is not None else ""

    if not channel_name:
        if ch_str.isdigit():
            channel_name = os.getenv("PPE_CHANNEL_NAME", ch_str)
        elif ch_str:
            channel_name = os.getenv("PPE_CHANNEL_NAME", ch_str)
        else:
            channel_name = os.getenv("PPE_CHANNEL_NAME", "CAM-UNKNOWN")

    if not device_sn:
        device_sn = (
            os.getenv("PPE_DEVICE_SN")
            or os.getenv("DEVICE_SN")
            or os.getenv("HOSTNAME")
            or "UNKNOWN"
        )

    cap_uuid = uuid.uuid4()
    scene_uuid = uuid.uuid4()
    correlation_id = uuid.uuid4().hex
    capture_id = f"{ALGORITHM_TYPE}_{cap_uuid}.jpg"
    scene_id = f"{ALGORITHM_TYPE}_{scene_uuid}.jpg"

    now = datetime.now(timezone.utc)
    if record_ms is None:
        record_ms = int(now.timestamp() * 1000)
    date_utc = datetime.fromtimestamp(
        record_ms / 1000, tz=timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    cloud = (os.getenv("PPE_CLOUD_IMAGE_BASE") or "").strip().rstrip("/")
    if cloud:
        cap_base = cloud
        scene_base = cloud
    else:
        cap_base = (os.getenv("PPE_CAPTURE_URL_BASE") or "").strip().rstrip("/")
        scene_base = (os.getenv("PPE_SCENE_URL_BASE") or "").strip().rstrip("/")

    local_fallback = "file:///local/storage/images"
    if _env_truthy("PPE_FORCE_LOCAL_URLS"):
        if not cap_base:
            cap_base = local_fallback
        if not scene_base:
            scene_base = local_fallback

    capture_url = f"{cap_base}/{capture_id}{correlation_id}.jpg" if cap_base else ""
    scene_url = f"{scene_base}/{scene_id}{correlation_id}.jpg" if scene_base else ""

    date_folder = now.strftime("%Y-%m-%d")
    return {
        "algorithmType"   : ALGORITHM_TYPE,
        "captureId"       : capture_id,
        "sceneId"         : scene_id,
        "channelId"       : ch_out,
        "channelName"     : channel_name,
        "deviceSN"        : device_sn,
        "id"              : correlation_id,
        "taskId"          : int(task_id),
        "taskName"        : task_name,
        "recordTime"      : record_ms,
        "dateUTC"         : date_utc,
        "personStructural": build_ppe_person_structural(
            alarm_type, area_points, bbox, score
        ),
        "captureUrl"      : capture_url,
        "sceneUrl"        : scene_url,
        "evidence"        : {
            "captureImage": build_image(f"{date_folder}/{capture_id}", "capture"),
            "sceneImage"  : build_image(f"{date_folder}/{scene_id}", "scene"),
        },
    }


# ── Polygon helpers ───────────────────────────────────────────────────────────

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

        self.channel_name = str(
            task_config.get("channelName")
            or detail.get("channelName")
            or ""
        ).strip()
        self.device_sn = str(
            task_config.get("deviceSN")
            or detail.get("deviceSN")
            or ""
        ).strip()

        self.zones = self._parse_zones(task_config.get("areaPosition", "[]"))

        raw_days            = task_config.get("validWeekday", list(_WEEKDAY_MAP.keys()))
        self.valid_weekdays = {_WEEKDAY_MAP[d] for d in raw_days if d in _WEEKDAY_MAP}
        self.valid_start_ms = task_config.get("validStartTime", 0)
        self.valid_end_ms   = task_config.get("validEndTime",   86400000)

        from services.ppe import PPEService
        from collections import deque as _deque
        self._ppe = PPEService()

        # QW-9: temporal voting window — suppress jittery per-frame alarms.
        # PPE_VOTE_WINDOW=1 (default) keeps the per-frame behavior unchanged.
        try:
            self._vote_window = max(1, int(os.getenv("PPE_VOTE_WINDOW", "1")))
        except ValueError:
            self._vote_window = 1
        try:
            self._vote_threshold = min(1.0, max(0.0, float(os.getenv("PPE_VOTE_THRESHOLD", "0.6"))))
        except ValueError:
            self._vote_threshold = 0.6
        # {(track_id, alarm_type): deque of bool} — True = non-compliant in that frame
        self._vote_history: dict = {}

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
            f"[MaskHairnetChefHat/{self.task_id}] Ready — "
            f"alarmTypes={self.alarm_types}, zones={len(self.zones)}, "
            f"vote_window={self._vote_window}"
        )

    # ── Main entry point ──────────────────────────────────────────────────────

    def __call__(self, payload: dict) -> list:
        if not self.enable or not self.alarm_types or not self._in_schedule():
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
                non_compliant = required_class not in detected_classes

                if self._vote_window > 1:
                    from collections import deque as _deque
                    key = (det.track_id, alarm_type)
                    hist = self._vote_history.get(key)
                    if hist is None:
                        hist = _deque(maxlen=self._vote_window)
                        self._vote_history[key] = hist
                    hist.append(non_compliant)
                    # Fire only when the majority of the window is non-compliant.
                    non_compliant = (
                        len(hist) >= self._vote_window
                        and sum(hist) / len(hist) >= self._vote_threshold
                    )

                if non_compliant:
                    # Use threshold as fallback confidence when class absent
                    conf_pct = int(ppe_conf_map.get(required_class, self.threshold) * 100)
                    zone     = self.zones[0] if self.zones else None
                    event = self._build_event(
                        det,
                        alarm_type,
                        conf_pct,
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
        alarm_type: str,
        conf_pct: int,
        zone: Optional[dict],
        timestamp: str,
        camera_id: str,
    ) -> dict:
        now_ms   = int(time.time() * 1000)
        event_id = hashlib.md5(
            f"{self.task_id}_{det.track_id}_{alarm_type}_{now_ms}".encode()
        ).hexdigest()

        x1, y1, x2, y2 = det.bbox
        area_points     = zone.get("point", []) if zone else []
        cam_key         = str(camera_id or self.channel_id or "unknown")

        data = build_ppe_spec_data(
            alarm_type=alarm_type,
            area_points=area_points,
            bbox=(x1, y1, x2, y2),
            score=conf_pct,
            task_id=self.task_id,
            task_name=self.task_name,
            channel_id=cam_key,
            channel_name=self.channel_name,
            device_sn=self.device_sn,
            record_ms=now_ms,
        )

        ch_top = str(cam_key)
        try:
            task_id_out: Any = int(self.task_id)
        except (TypeError, ValueError):
            task_id_out = self.task_id

        return {
            "eventId"     : event_id,
            "eventType"   : ALGORITHM_TYPE,
            "timestamp"   : now_ms,
            "timestampUTC": data["dateUTC"],
            "taskId"      : task_id_out,
            "taskName"    : self.task_name,
            "channelId"   : ch_top,
            "camera_id"   : camera_id or cam_key,
            "data"        : data,
            "evidence"    : data["evidence"],
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

        ps_raw = event.get("data", {}).get("personStructural", "{}")
        try:
            ps = json.loads(ps_raw)
        except json.JSONDecodeError:
            ps = {}
        alarm_type = ps.get("alarmType", "")
        area_raw = ps.get("areaPoints", "[]")
        if isinstance(area_raw, str):
            try:
                zone_points = json.loads(area_raw)
            except json.JSONDecodeError:
                zone_points = []
        else:
            zone_points = area_raw or []
        scene_vis = draw_evidence_scene(
            frame,
            subject_bbox=det.bbox,
            label=f"{alarm_type} id{det.track_id}",
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
            parsed = json.loads(area_position) if area_position else []
        except Exception as e:
            print(f"[MaskHairnetChefHat] Failed to parse areaPosition: {e}")
            return []
        if not parsed or not isinstance(parsed, list):
            return []
        # Bare polygon: [{"x":..,"y":..}, ...] → single zone
        first = parsed[0]
        if isinstance(first, dict) and "x" in first and "y" in first and "point" not in first:
            return [{"point": parsed}]
        return parsed
