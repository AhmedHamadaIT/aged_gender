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
    "channelId"     : str | int,    # JSON number accepted; stored as str
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

Environment (optional tuning):
- CROSSLINE_ANCHOR_POINT=bottom|center — bottom uses bbox bottom-center (feet) for
  line-side tests; better for doorway / floor lines in crowds. Default: bottom.
- CROSSLINE_SIDE_STATE_TTL_FRAMES=N — keep per-track line-side memory for N frames
  after the track last appeared (occlusion / missed detections). Default: 90.
  Set to 0 to restore immediate purge when a track is absent from a frame.
"""

import os
import json
import hashlib
import time
from datetime import datetime, timezone
from typing import Optional, Dict, Tuple, List, Set

import cv2

from utils import build_image, draw_evidence_scene, make_evidence_paths

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
        # Last frame_id each track was seen (FrameBus payload). Retain side state across
        # short occlusions / missed detections so crossings are not lost or double-counted.
        self._track_last_seen: Dict[int, int] = {}
        self._side_state_ttl_frames = max(
            0, int(os.getenv("CROSSLINE_SIDE_STATE_TTL_FRAMES", "90"))
        )
        self._fallback_frame_seq = 0
        _anchor = os.getenv("CROSSLINE_ANCHOR_POINT", "bottom").strip().lower()
        self._use_bottom_anchor = _anchor in ("bottom", "foot", "feet")

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
            f"{len(self.lines)} line(s), attr={self.enable_attr}, "
            f"anchor={'bottom' if self._use_bottom_anchor else 'center'}, "
            f"side_ttl_frames={self._side_state_ttl_frames}"
        )

    # ── Main entry point ──────────────────────────────────────────────────────

    def __call__(self, payload: dict) -> list:
        if not self.enable or not self.lines or not self._in_schedule():
            return []

        detection = payload["detection"]
        frame_cache = None

        def _frame_bgr():
            nonlocal frame_cache
            if frame_cache is None:
                from utils.task_payload import task_frame_bgr

                frame_cache = task_frame_bgr(payload)
            return frame_cache

        persons: List = [
            d for d in detection.get("items", [])
            if d.class_name == "person"
            and d.confidence >= self.threshold
            and d.track_id != -1            # skip detections with no track yet
        ]

        events: List = []
        active_track_ids: Set[int] = set()
        raw_fid = payload.get("frame_id")
        if raw_fid is None:
            self._fallback_frame_seq += 1
            frame_id = self._fallback_frame_seq
        else:
            try:
                frame_id = int(raw_fid)
            except (TypeError, ValueError):
                self._fallback_frame_seq += 1
                frame_id = self._fallback_frame_seq

        for det in persons:
            track_id = det.track_id
            active_track_ids.add(track_id)
            self._track_last_seen[track_id] = frame_id

            if track_id not in self._track_sides:
                self._track_sides[track_id] = {}

            anchor = self._crossing_anchor(det)
            for line in self.lines:
                crossing_dir = self._check_crossing(track_id, anchor, line)
                if crossing_dir is None:
                    continue

                attrs = self._get_attributes(
                    _frame_bgr() if self.enable_attr else None,
                    det,
                )
                event = self._build_event(
                    det,
                    line,
                    crossing_dir,
                    attrs,
                    payload["timestamp"],
                    str(payload.get("camera_id") or ""),
                )
                self._persist(event, _frame_bgr(), det, line, crossing_dir)
                events.append(event)

        self._purge_stale_track_state(frame_id, active_track_ids)

        return events

    # ── Line crossing ─────────────────────────────────────────────────────────

    def _check_crossing(self, track_id: int, point: Tuple[int, int], line: dict) -> Optional[int]:
        p1  = (line["point"][0]["x"], line["point"][0]["y"])
        p2  = (line["point"][1]["x"], line["point"][1]["y"])
        lid = line["line_id"]

        new_side  = _line_side(point, p1, p2)
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

    def _crossing_anchor(self, det) -> Tuple[int, int]:
        """Point used for line-side tests: foot/bottom-mid for counting, else bbox center."""
        if self._use_bottom_anchor:
            return ((det.x1 + det.x2) // 2, int(det.y2))
        return det.center

    def _purge_stale_track_state(self, frame_id: int, active_track_ids: Set[int]) -> None:
        if self._side_state_ttl_frames <= 0:
            # Legacy behaviour: drop side memory as soon as the track is absent.
            self._track_sides = {
                k: v for k, v in self._track_sides.items() if k in active_track_ids
            }
            self._track_last_seen = {
                k: v for k, v in self._track_last_seen.items() if k in active_track_ids
            }
            return

        ttl = self._side_state_ttl_frames
        for tid in list(self._track_sides.keys()):
            if tid in active_track_ids:
                continue
            last = self._track_last_seen.get(tid)
            if last is None:
                self._track_sides.pop(tid, None)
                continue
            if frame_id - last > ttl:
                self._track_sides.pop(tid, None)
                self._track_last_seen.pop(tid, None)

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

    def _build_event(
        self,
        det,
        line: dict,
        crossing_dir: int,
        attrs: dict,
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

        return {
            "eventId"     : event_id,
            "eventType"   : "CROSS_LINE",
            "timestamp"   : now_ms,
            "timestampUTC": datetime.fromtimestamp(
                now_ms / 1000, tz=timezone.utc
            ).isoformat().replace("+00:00", "Z"),
            "taskId"      : self.task_id,
            "taskName"    : self.task_name,
            "channelId"   : str(camera_id or self.channel_id),
            "camera_id"   : camera_id,
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
                "captureImage": build_image(cap_rel, "capture"),
                "sceneImage"  : build_image(scene_rel, "scene"),
            },
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def _persist(self, event: dict, frame, det, line: dict, crossing_dir: int):
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
        pts = line["point"]
        scene_vis = draw_evidence_scene(
            frame,
            subject_bbox=det.bbox,
            label=f"{line.get('line_name') or line.get('line_id','')} dir{crossing_dir} id{det.track_id}",
            line_endpoints=(
                (int(pts[0]["x"]), int(pts[0]["y"])),
                (int(pts[1]["x"]), int(pts[1]["y"])),
            ),
        )
        cv2.imwrite(scene_path, scene_vis)

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
