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

import logging
import os
import json
import hashlib
import time
from datetime import datetime, timezone
from typing import Any, Optional, Dict, Tuple, List, Set

import cv2

_log_cl = logging.getLogger(__name__)

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


def parse_effective_cross_lines(area_position: Any) -> List[dict]:
    """
    Normalized line definitions used by ``CrossLineTask`` and the live-stream
    geometry overlay. Entries without two valid ``{x,y}`` points are dropped.
    ``line_id`` defaults from ``line_name`` or a stable ``line_<index>`` so the
    worker and preview never disagree on which segments exist.
    """
    if area_position is None:
        return []
    try:
        if isinstance(area_position, list):
            arr = area_position
        else:
            s = str(area_position).strip()
            if not s:
                return []
            arr = json.loads(s)
    except (json.JSONDecodeError, TypeError, ValueError):
        return []
    if not isinstance(arr, list):
        return []
    out: List[dict] = []
    for i, obj in enumerate(arr):
        if not isinstance(obj, dict):
            continue
        pts = obj.get("point")
        if not isinstance(pts, list) or len(pts) < 2:
            continue
        a, b = pts[0], pts[1]
        if not isinstance(a, dict) or not isinstance(b, dict):
            continue
        try:
            x0, y0 = int(a["x"]), int(a["y"])
            x1, y1 = int(b["x"]), int(b["y"])
        except (KeyError, TypeError, ValueError):
            continue
        lid_raw = obj.get("line_id")
        if isinstance(lid_raw, str):
            lid_raw = lid_raw.strip()
        if lid_raw is None or lid_raw == "":
            ln = obj.get("line_name")
            if isinstance(ln, str) and ln.strip():
                lid_raw = ln.strip()
            else:
                lid_raw = f"line_{i}"
        else:
            lid_raw = str(lid_raw)
        raw_dir = obj.get("direction", 0)
        try:
            direction_cfg = int(raw_dir)
        except (TypeError, ValueError):
            direction_cfg = 0
        if direction_cfg not in (0, 1, 2):
            direction_cfg = 0
        out.append(
            {
                "line_id": lid_raw,
                "line_name": str(obj.get("line_name") or ""),
                "point": [{"x": x0, "y": y0}, {"x": x1, "y": y1}],
                "direction": direction_cfg,
            }
        )
    return out


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

        self.lines = parse_effective_cross_lines(task_config.get("areaPosition", "[]"))

        # ── Init-time config validation (step 9 safety net) ──────────────────
        # The API-time validator already checks this, but we re-validate here so
        # a worker process started with a stale task config fails loudly at init
        # rather than silently on the first frame.
        if not self.lines and self.enable:
            raise ValueError(
                f"CrossLineTask {self.task_id} has no valid lines in areaPosition. "
                f"Provide a JSON array with at least one line (two {{x,y}} points)."
            )

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

        # Reentry grace: when a track reappears after being absent for ≥ this
        # many frames, its previous side state is preserved but a crossing is
        # NOT immediately fired even if the new position is on the opposite side.
        # This prevents false positives on occlusion re-entry.
        # Set CROSSLINE_REENTRY_GRACE_FRAMES=0 to disable.
        self._reentry_grace_frames = max(
            0, int(os.getenv("CROSSLINE_REENTRY_GRACE_FRAMES", "5"))
        )
        # {track_id: frame_id_when_track_returned} — tracks under grace period
        self._reentry_grace_active: Dict[int, int] = {}

        # Crowded scene throttle: when more than this many person detections
        # arrive in a single frame, keep only the top-N by confidence to avoid
        # O(N²) crossing checks and reduce ID instability from low-conf ghosts.
        # Set CROSSLINE_MAX_TRACKS_PER_FRAME=0 to disable.
        self._max_tracks_per_frame = max(
            0, int(os.getenv("CROSSLINE_MAX_TRACKS_PER_FRAME", "0"))
        )

        self._fallback_frame_seq = 0
        _anchor = os.getenv("CROSSLINE_ANCHOR_POINT", "bottom").strip().lower()
        self._use_bottom_anchor = _anchor in ("bottom", "foot", "feet")

        # Age/Gender — loaded only when enableAttrDetect is true
        self._age_gender = None
        self._age_gender_load_error: Optional[str] = None
        if self.enable_attr:
            try:
                from services.age_gender import AgeGenderService

                self._age_gender = AgeGenderService()
            except Exception as e:
                self._age_gender_load_error = str(e)
                _log_cl.warning(
                    "[CrossLine/%s] AgeGenderService disabled: %s",
                    self.task_id, e,
                )

        # Local storage
        self._capture_dir = os.getenv("CAPTURE_DIR", "/local/storage/captures")
        self._scene_dir   = os.getenv("SCENE_DIR",   "/local/storage/scenes")
        self._events_dir  = os.getenv("EVENTS_DIR",  "/local/storage/events")
        os.makedirs(self._capture_dir, exist_ok=True)
        os.makedirs(self._scene_dir,   exist_ok=True)
        os.makedirs(self._events_dir,  exist_ok=True)

        self._jsonl_path = os.path.join(self._events_dir, f"task_{self.task_id}.jsonl")

        _log_cl.info(
            "[CrossLine/%s] Ready — lines=%d attr=%s model=%s anchor=%s "
            "side_ttl=%d reentry_grace=%d max_tracks=%s",
            self.task_id,
            len(self.lines),
            self.enable_attr,
            "loaded" if (self.enable_attr and self._age_gender) else (
                "FAILED:" + self._age_gender_load_error if self._age_gender_load_error else "n/a"
            ),
            "bottom" if self._use_bottom_anchor else "center",
            self._side_state_ttl_frames,
            self._reentry_grace_frames,
            self._max_tracks_per_frame if self._max_tracks_per_frame > 0 else "unlimited",
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

        # Crowded scene throttle: keep only top-N highest-confidence tracks
        if self._max_tracks_per_frame > 0 and len(persons) > self._max_tracks_per_frame:
            persons = sorted(persons, key=lambda d: d.confidence, reverse=True)[
                : self._max_tracks_per_frame
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

            # Reentry grace: detect tracks that were absent and just reappeared.
            last_seen = self._track_last_seen.get(track_id)
            if (
                last_seen is not None
                and self._reentry_grace_frames > 0
                and frame_id - last_seen > self._reentry_grace_frames
            ):
                # Track re-entered after an absence longer than the grace window —
                # record the grace start so crossings are suppressed for the next
                # _reentry_grace_frames frames while the side state settles.
                self._reentry_grace_active[track_id] = frame_id

            self._track_last_seen[track_id] = frame_id

            if track_id not in self._track_sides:
                self._track_sides[track_id] = {}

            anchor = self._crossing_anchor(det)

            # Determine whether this track is in the reentry grace window.
            grace_start = self._reentry_grace_active.get(track_id)
            in_grace = (
                grace_start is not None
                and self._reentry_grace_frames > 0
                and frame_id - grace_start < self._reentry_grace_frames
            )
            if grace_start is not None and not in_grace:
                # Grace period expired — remove the marker.
                self._reentry_grace_active.pop(track_id, None)

            for line in self.lines:
                crossing_dir = self._check_crossing(track_id, anchor, line)
                if crossing_dir is None:
                    continue
                if in_grace:
                    # Update side state but do NOT fire crossing event — the
                    # track is still settling after reappearing from occlusion.
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
            self._reentry_grace_active = {
                k: v for k, v in self._reentry_grace_active.items() if k in active_track_ids
            }
            return

        ttl = self._side_state_ttl_frames
        for tid in list(self._track_sides.keys()):
            if tid in active_track_ids:
                continue
            last = self._track_last_seen.get(tid)
            if last is None:
                self._track_sides.pop(tid, None)
                self._reentry_grace_active.pop(tid, None)
                continue
            if frame_id - last > ttl:
                self._track_sides.pop(tid, None)
                self._track_last_seen.pop(tid, None)
                self._reentry_grace_active.pop(tid, None)

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
