"""
services/cashier.py  —  v1.2.0

CashierService — ROI-based cashier drawer monitor.

CashierDrawerTask — FrameBus task worker for ``algorithmType`` ``CASHIER_DRAWER``
(``POST /api/tasks``). Adapts each FrameBus ``payload`` to the pipeline-style
``context`` dict and invokes ``CashierService``. Same file keeps service + task
adapter in one place.

Model   : YOLO11m  (best_cashier.onnx)
Classes : 0=Person  1=Drawer_Open  2=Cash
Zones   : ROI_CASHIER (staff side)  •  ROI_CUSTOMER (customer side)

Reads   context["data"]["detection"]["items"]   — List[Detection]
        context["data"]["frame"]                — np.ndarray BGR

Writes  context["data"]["use_case"]["cashier"]  — dict (Hybrid schema)
        {
          "data": { ... },                     # Eyego spec §4 top-level payload
          "case_id" / "severity"               # flat copies for API filters
          "persons": [                          # person-level (like PPE)
            {
              "person_bbox" : List[int],
              "confidence"  : float,
              "zone"        : str,              # ROI_CASHIER | ROI_CUSTOMER
              "transaction" : bool,
              "items": {
                "drawers": [{"bbox": List[int], "confidence": float|None}],
                "cash"   : [{"bbox": List[int], "confidence": float|None}],
              }
            }
          ],
          "summary": {                          # frame-level (legacy)
            "cashier_zone"  : { "persons": int, "drawers": int, "cash": int },
            "customer_zone" : { "persons": int, "drawers": int, "cash": int },
            "case_id"       : str,              # N1–N6 / A1–A7
            "severity"      : str,              # NORMAL | ALERT | CRITICAL
            "alerts"        : List[str],
            "transaction"   : bool,
            "frame_saved"   : bool,
            "evidence_path" : str | None,
            "frame_id"      : int,
            "timestamp"     : str,
            "cashier_persons": List[dict],
          }
        }

        context["data"]["detection"]["items"]   — filtered to persons only (class_id == 0)
        context["data"]["detection"]["count"]   — updated count

Environment variables

CASHIER_MODEL            : ONNX model path         default: ./models/best_cashier.onnx
CASHIER_CONFIG           : zone config (YAML/JSON)  default: ./config/cashier_zones.yaml
CASHIER_EVIDENCE_DIR     : evidence output dir      default: ./evidence/cashier
CASHIER_CAMERA_ID        : camera label in files    default: cam
CASHIER_CONF_PERSON      : Person confidence        default: 0.50
CASHIER_CONF_DRAWER      : Drawer_Open confidence   default: 0.45
CASHIER_CONF_CASH        : Cash confidence          default: 0.50
CASHIER_DRAWER_MAX_SEC   : seconds → A6 escalation  default: 30
CASHIER_WAIT_MAX_SEC     : seconds → A5 escalation  default: 30
CASHIER_PROXIMITY_IOU    : nearby IoU threshold      default: 0.05
CASHIER_RELOAD_INTERVAL  : config reload (seconds)  default: 60
CASHIER_BUFFER_SIZE      : rolling frame buffer      default: 100
SAVE_OUTPUT              : annotate + save frames   default: True

Integration envelope (spec §4 ``data`` block), optional env:

CASHIER_CLOUD_IMAGE_BASE : prefix for captureUrl/sceneUrl (both if capture/scene unset)
CASHIER_CAPTURE_URL_BASE : override captureUrl base only
CASHIER_SCENE_URL_BASE   : override sceneUrl base only
CASHIER_DEVICE_SN        : deviceSN in ``data`` (else DEVICE_SN, else UNKNOWN)
CASHIER_CHANNEL_NAME     : channelName if not in config task

CASHIER_DISABLE_DRAWER_TOTAL_PERSIST : set 1 to skip JSON persistence (RAM-only total)
CASHIER_DRAWER_DURATION_PERSIST_SEC  : wall-clock min interval to flush duration while
                                       drawer stays open (default 10); always flush on open/close edge

Optional staff bindings (integration; not set by pipeline today):

    context["data"]["cashier_staff"] = mock_cashier_staff_context([
        {"bbox": [x1,y1,x2,y2], "staff_id": 1001},
    ])

If detail_config.enable_staff_list is true and bindings are missing/empty, every
cashier-zone person is treated as unauthorized (strict).

Additive use_case.cashier keys: personStructural (JSON string), algorithmType.
summary.* keys are unchanged.

personStructural includes:
  total_open_count — rising-edge opens (closed → open) in cashier ROI, per camera.
  total_open_duration_ms — cumulative ms the drawer was open (inter-frame dt while
    previous sample was open); persisted with totals file.
  current_open_duration_ms — ms since current open streak started (0 if closed);
    sent every frame. Legacy drawer_open_duration_ms kept for A1/A3/A4/A6 when >0.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from dotenv import load_dotenv

from logger.logger_config import Logger

# Optional GIF support
try:
    import imageio
    _HAS_IMAGEIO = True
except ImportError:
    _HAS_IMAGEIO = False

load_dotenv()
log = Logger.get_logger(__name__)

STAFF_CONTEXT_KEY = "cashier_staff"
ALGORITHM_TYPE = "CASHIER_BOX_OPEN"
_CRITICAL_UNBOUND_WARN_SEC = 600.0
_STAFF_MATCH_IOU = 0.25
# Persistent cumulative drawer-open edges (closed → open), per camera_id
CASHIER_DRAWER_TOTALS_FILENAME = "cashier_drawer_open_totals.json"

# ─────────────────────────────────────────────
# Zone identifiers
# ─────────────────────────────────────────────
ZONE_CASHIER  = "ROI_CASHIER"
ZONE_CUSTOMER = "ROI_CUSTOMER"
ZONE_OUTSIDE  = "OUTSIDE"

# ─────────────────────────────────────────────
# Severity levels
# ─────────────────────────────────────────────
SEVERITY_NORMAL   = "NORMAL"
SEVERITY_ALERT    = "ALERT"
SEVERITY_CRITICAL = "CRITICAL"

# Task config ``algorithmType`` (POST /api/tasks) — must match apis/tasks.TaskRegistry
ALGORITHM_TYPE = "CASHIER_DRAWER"

# ─────────────────────────────────────────────
# Draw colours (BGR)
# ─────────────────────────────────────────────
_CLR_CASHIER  = (0,  200, 100)   # green
_CLR_CUSTOMER = (0,  165, 255)   # orange
_CLR_ALERT    = (0,  165, 255)   # amber
_CLR_CRITICAL = (0,    0, 255)   # red
_CLR_NORMAL   = (200, 200, 200)  # grey

# ─────────────────────────────────────────────
# Case → evidence sub-folder
# ─────────────────────────────────────────────
_CASE_FOLDER: Dict[str, str] = {
    "N3": "normal/N3",
    "A1": "abnormal/A1_unattended",
    "A2": "abnormal/A2_intrusion",
    "A3": "abnormal/A3_critical",
    "A4": "abnormal/A4_critical",
    "A5": "abnormal/A5_service",
    "A6": "abnormal/A6_extended",
    "A7": "abnormal/A7_cash_kz",
}

# ─────────────────────────────────────────────
# Debounce: consecutive frames required to fire
# ─────────────────────────────────────────────
_DEBOUNCE_MAP: Dict[str, int] = {"A3": 1, "A4": 1}
_DEBOUNCE_DEFAULT = 3

# ─────────────────────────────────────────────
# GIF budgets per case
# ─────────────────────────────────────────────
# max_post None = uncapped until case resolves (A3/A4); see _accumulate_post
_GIF_BUDGET: Dict[str, Dict] = {
    "A1": {"pre": 30,  "max_post": 600,  "fps": 10, "keyframe_interval": None},
    "A2": {"pre": 50,  "max_post": 30,   "fps": 10, "keyframe_interval": None},
    "A3": {"pre": 60,  "max_post": None, "fps": 10, "keyframe_interval": None},
    "A4": {"pre": 60,  "max_post": None, "fps": 10, "keyframe_interval": None},
    "A5": {"pre": 0,   "max_post": 60,   "fps": 1,  "keyframe_interval": 30.0},
    "A6": {"pre": 0,   "max_post": 60,   "fps": 1,  "keyframe_interval": 10.0},
    "A7": {"pre": 20,  "max_post": 200,  "fps": 5,  "keyframe_interval": None},
}


# ─────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────

def _load_config(path: str) -> Dict:
    """Load zone config from YAML or JSON. Falls back to default centred zones."""
    try:
        if path.endswith((".yaml", ".yml")):
            import yaml
            with open(path) as f:
                return yaml.safe_load(f) or {}
        with open(path) as f:
            return json.load(f)
    except Exception as exc:
        log.warning("[CASHIER] Config load failed (%s) — using default zones", exc)
        return _default_config()


def _default_config() -> Dict:
    """Fallback: left half = cashier zone, right half = customer zone."""
    return {
        "meta": {"version": "1.2.0"},
        "zones": {
            ZONE_CASHIER : {"shape": "rectangle", "points": [[0.0, 0.0], [0.5, 1.0]], "active": True},
            ZONE_CUSTOMER: {"shape": "rectangle", "points": [[0.5, 0.0], [1.0, 1.0]], "active": True},
        },
        "thresholds": {
            "drawer_open_max_seconds"  : 30,
            "customer_wait_max_seconds": 30,
            "proximity_iou"            : 0.05,
            "config_reload_interval"   : 60,
        },
        "buffer"  : {"size": 100, "jpeg_quality": 75},
        "gif"     : {"fps": 10, "quality": 85},
        "debounce": {"default": 3, "A3": 1, "A4": 1},
        "evidence": {"save_gif": True, "save_thumbnail": True, "log_rotate_mb": 100},
        "detail_config": {
            "enable_staff_list": False,
            "staff_ids": [],
        },
    }


def mock_cashier_staff_context(bindings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build context[\"data\"][\"cashier_staff\"] for tests or pre-integration.
    Each binding: {\"bbox\": [x1,y1,x2,y2], \"staff_id\": int}
    """
    return {"bindings": list(bindings)}


# ─────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────

def _iou(b1: List[float], b2: List[float]) -> float:
    """Intersection-over-Union for two [x1,y1,x2,y2] boxes."""
    xi1 = max(b1[0], b2[0]);  yi1 = max(b1[1], b2[1])
    xi2 = min(b1[2], b2[2]);  yi2 = min(b1[3], b2[3])
    inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
    union = (b1[2] - b1[0]) * (b1[3] - b1[1]) + (b2[2] - b2[0]) * (b2[3] - b2[1]) - inter
    return inter / union if union > 0 else 0.0


def _point_in_poly(px: float, py: float, poly: List[List[float]]) -> bool:
    """Ray-casting point-in-polygon test. poly = list of [x,y] normalised coords."""
    n, inside, j = len(poly), False, len(poly) - 1
    for i in range(n):
        xi, yi = poly[i];  xj, yj = poly[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-9) + xi):
            inside = not inside
        j = i
    return inside


def _rect_to_poly(pts: List[List[float]]) -> List[List[float]]:
    """Convert [[x1,y1],[x2,y2]] rectangle spec to 4-corner polygon."""
    (x1, y1), (x2, y2) = pts[0], pts[1]
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def _denorm_poly(poly: List[List[float]], w: int, h: int) -> List[Tuple[int, int]]:
    return [(int(x * w), int(y * h)) for x, y in poly]


def _build_poly(zone_cfg: Dict) -> List[List[float]]:
    """Build a normalised polygon from zone config, respecting the active flag."""
    if zone_cfg.get("active") is False:
        return []
    pts   = zone_cfg.get("points", [])
    if not pts:
        return []
    shape = zone_cfg.get("shape", "polygon")
    return _rect_to_poly(pts) if shape == "rectangle" and len(pts) == 2 else pts


# ─────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────

@dataclass
class ZoneCount:
    persons: int = 0
    drawers: int = 0
    cash   : int = 0
    person_bboxes      : List[List[float]] = field(default_factory=list)
    drawer_bboxes      : List[List[float]] = field(default_factory=list)
    cash_bboxes        : List[List[float]] = field(default_factory=list)
    person_confidences : List[float]       = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {"persons": self.persons, "drawers": self.drawers, "cash": self.cash}


@dataclass
class PersonEntry:
    """
    Person-level entry — mirrors PPE schema.
    cash + drawers linked via drawer-proxy IoU (person→drawer→cash).
    """
    person_bbox : List[float]
    confidence  : float
    zone        : str
    transaction : bool = False
    items       : Dict[str, List[Dict]] = field(default_factory=lambda: {
        "drawers": [],
        "cash"   : [],
    })

    def to_dict(self) -> Dict:
        return {
            "person_bbox": [int(c) for c in self.person_bbox],
            "confidence" : round(self.confidence, 4),
            "zone"       : self.zone,
            "transaction": self.transaction,
            "items"      : self.items,
        }


@dataclass
class CashierResult:
    cashier_zone  : ZoneCount
    customer_zone : ZoneCount
    case_id       : str
    severity      : str
    alerts        : List[str]
    transaction   : bool
    frame_saved   : bool
    evidence_path : Optional[str]
    persons       : List[PersonEntry] = field(default_factory=list)   # ← person-level
    frame_id      : int = 0
    timestamp     : str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    person_structural     : Optional[str] = None
    unauthorized_present  : bool = False
    drawer_open_duration_ms: Optional[int] = None
    wait_duration_ms      : Optional[int] = None
    task_meta             : Optional[Dict[str, Any]] = None
    total_open_count         : int = 0  # cumulative drawer open edges (CASHIER zone)
    total_open_duration_ms   : int = 0  # cumulative ms drawer was open (sampled inter-frame)
    current_open_duration_ms : int = 0  # ms in current open streak; 0 if closed

    def to_dict(self) -> Dict:
        # Legacy cashier_persons list (kept for backward-compat)
        cashier_persons = [
            {
                "bbox"      : [int(c) for c in bbox],
                "confidence": conf,
                "zone"      : ZONE_CASHIER,
            }
            for bbox, conf in zip(
                self.cashier_zone.person_bboxes,
                self.cashier_zone.person_confidences,
            )
        ]
        out: Dict[str, Any] = {
            "persons": [p.to_dict() for p in self.persons],
            "summary": {
                "cashier_zone"   : self.cashier_zone.to_dict(),
                "customer_zone"  : self.customer_zone.to_dict(),
                "case_id"        : self.case_id,
                "severity"       : self.severity,
                "alerts"         : self.alerts,
                "transaction"    : self.transaction,
                "frame_saved"    : self.frame_saved,
                "evidence_path"  : self.evidence_path,
                "frame_id"       : self.frame_id,
                "timestamp"      : self.timestamp,
                "cashier_persons": cashier_persons,
                "total_open_count": self.total_open_count,
                "total_open_duration_ms": self.total_open_duration_ms,
                "current_open_duration_ms": self.current_open_duration_ms,
            },
            "algorithmType": ALGORITHM_TYPE,
        }
        if self.person_structural is not None:
            out["personStructural"] = self.person_structural
        if self.task_meta:
            out["task"] = dict(self.task_meta)
        # Flat fields for API filters / SSE consumers
        out["case_id"] = self.case_id
        out["severity"] = self.severity
        # Eyego-style integration payload (same keys as spec §4 ``data``)
        out["data"] = build_cashier_spec_data(self)
        return out


def build_cashier_spec_data(result: CashierResult) -> Dict[str, Any]:
    """
    Build the backend ``data`` object (algorithmType, captureId, personStructural, …).
    URLs are optional; set CASHIER_CLOUD_IMAGE_BASE or per-field *_URL_BASE env vars.
    GIF / frame-save logic is unchanged — this only shapes JSON for consumers.
    """
    meta = result.task_meta if isinstance(result.task_meta, dict) else {}

    def _pick_int(*keys: str, default: int = 0) -> int:
        for k in keys:
            if k in meta and meta[k] is not None:
                try:
                    return int(meta[k])
                except (TypeError, ValueError):
                    return default
        return default

    def _pick_str(*keys: str, default: str = "") -> str:
        for k in keys:
            v = meta.get(k)
            if v is not None and str(v).strip() != "":
                return str(v)
        return default

    channel_id = _pick_int("channelId", "channel_id")
    task_id = meta.get("taskId", meta.get("task_id"))
    task_id_out: Any
    if task_id is None or task_id == "":
        task_id_out = None
    else:
        try:
            task_id_out = int(task_id)
        except (TypeError, ValueError):
            task_id_out = None

    channel_name = _pick_str("channelName", "channel_name")
    if not channel_name:
        if channel_id:
            channel_name = os.getenv(
                "CASHIER_CHANNEL_NAME",
                f"CAM-{channel_id:02d}-MAIN",
            )
        else:
            channel_name = os.getenv("CASHIER_CHANNEL_NAME", "CAM-UNKNOWN")

    device_sn = _pick_str("deviceSN", "device_sn")
    if not device_sn:
        device_sn = (
            os.getenv("CASHIER_DEVICE_SN")
            or os.getenv("DEVICE_SN")
            or "UNKNOWN"
        )

    algo = _pick_str("algorithmType", "algorithm_type") or ALGORITHM_TYPE
    capture_id = f"CASHIER_BOX_OPEN_{uuid.uuid4()}.jpg"
    scene_id = f"CASHIER_BOX_OPEN_{uuid.uuid4()}.jpg"

    now = datetime.now(timezone.utc)
    record_ms = int(now.timestamp() * 1000)
    date_utc = now.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    cap_base = (
        os.getenv("CASHIER_CAPTURE_URL_BASE")
        or os.getenv("CASHIER_CLOUD_IMAGE_BASE")
        or ""
    ).rstrip("/")
    scene_base = (
        os.getenv("CASHIER_SCENE_URL_BASE")
        or os.getenv("CASHIER_CLOUD_IMAGE_BASE")
        or ""
    ).rstrip("/")
    capture_url = f"{cap_base}/{capture_id}" if cap_base else ""
    scene_url = f"{scene_base}/{scene_id}" if scene_base else ""

    ps = result.person_structural if result.person_structural is not None else "{}"

    return {
        "algorithmType": algo,
        "captureId": capture_id,
        "sceneId": scene_id,
        "channelId": channel_id,
        "channelName": channel_name,
        "deviceSN": device_sn,
        "id": uuid.uuid4().hex,
        "taskId": task_id_out,
        "taskName": _pick_str("taskName", "task_name"),
        "recordTime": record_ms,
        "dateUTC": date_utc,
        "total_open_count": int(result.total_open_count),
        "total_open_duration_ms": int(result.total_open_duration_ms),
        "current_open_duration_ms": int(result.current_open_duration_ms),
        "personStructural": ps,
        "captureUrl": capture_url,
        "sceneUrl": scene_url,
    }


# ─────────────────────────────────────────────
# Evidence writer
# ─────────────────────────────────────────────

class _EvidenceWriter:
    """Thread-safe annotated-frame + JSON-metadata + JSONL-log saver."""

    def __init__(self, base_dir: str, log_rotate_mb: float = 100.0):
        self.base = Path(base_dir)
        self.base.mkdir(parents=True, exist_ok=True)
        self._log_path      = self.base / "logs" / "events.jsonl"
        self._log_rotate_mb = log_rotate_mb
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def case_folder(self, case_id: str) -> Path:
        sub    = _CASE_FOLDER.get(case_id, f"other/{case_id}")
        folder = self.base / sub
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def save_frame(
        self,
        frame    : np.ndarray,
        case_id  : str,
        meta     : Dict,
        camera_id: str = "cam",
        quality  : int = 85,
    ) -> str:
        """Save annotated JPEG + JSON metadata to the correct case folder."""
        folder    = self.case_folder(case_id)
        ts        = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
        img_path  = folder / f"{camera_id}_{ts}.jpg"
        meta_path = folder / f"{camera_id}_{ts}.json"
        with self._lock:
            cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            meta_path.write_text(json.dumps({**meta, "saved_at": ts}, indent=2))
        log.info("[CASHIER] Frame saved → %s", img_path)
        return str(img_path)

    def append_log(self, record: Dict) -> None:
        """Append one JSON line to events.jsonl. Rotates if over size limit."""
        with self._lock:
            self._rotate_if_needed()
            with self._log_path.open("a") as f:
                f.write(json.dumps(record) + "\n")

    def _rotate_if_needed(self) -> None:
        """Rotate log file when it exceeds the size limit (called under _lock)."""
        if not self._log_path.exists():
            return
        size_mb = self._log_path.stat().st_size / (1024 * 1024)
        if size_mb >= self._log_rotate_mb:
            ts       = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            new_name = self._log_path.with_name(f"events.{ts}.jsonl")
            self._log_path.rename(new_name)
            log.info("[CASHIER] Log rotated → %s", new_name)

    def compile_gif(
        self,
        frames   : List[bytes],
        case_id  : str,
        camera_id: str,
        fps      : int,
        quality  : int = 85,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Compile JPEG byte list into a GIF and thumbnail.
        Returns (gif_path, thumb_path) — either may be None on failure.
        """
        if not frames:
            return None, None

        folder     = self.case_folder(case_id)
        ts         = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        gif_path   = folder / f"{camera_id}_{ts}.gif"
        thumb_path = folder / f"{camera_id}_{ts}_thumb.jpg"

        first = cv2.imdecode(np.frombuffer(frames[0], np.uint8), cv2.IMREAD_COLOR)
        if first is not None:
            cv2.imwrite(str(thumb_path), first, [cv2.IMWRITE_JPEG_QUALITY, quality])

        if not _HAS_IMAGEIO:
            log.warning("[CASHIER] imageio not available — GIF skipped for %s", case_id)
            return None, str(thumb_path)

        try:
            rgb_frames = []
            for fb in frames:
                img = cv2.imdecode(np.frombuffer(fb, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    rgb_frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            if rgb_frames:
                imageio.mimsave(str(gif_path), rgb_frames, fps=fps, loop=0)
                log.info("[CASHIER] GIF saved → %s (%d frames)", gif_path, len(rgb_frames))
                return str(gif_path), str(thumb_path)
        except Exception as exc:
            log.warning("[CASHIER] GIF generation failed: %s", exc)

        return None, str(thumb_path)


# ─────────────────────────────────────────────
# CashierService
# ─────────────────────────────────────────────

class CashierService:
    """
    Pipeline service for ROI-based cashier drawer monitoring.

    Evaluated after DetectorService — expects context["data"]["detection"]["items"]
    to contain Detection objects produced by services/detector.py.

    Pipeline usage:
        POST /detection/setup
        {"pipeline": ["detector", "cashier"]}

    Output schema (v1.2.0):
        context["data"]["use_case"]["cashier"] = {
            "persons": [...],    # person-level, like PPE
            "summary": {...},    # frame-level, legacy
        }
        context["data"]["detection"]["items"]  = persons only (drawers/cash removed)
        context["data"]["detection"]["count"]  = updated
    """

    _CONF: Dict[int, float] = {
        0: float(os.getenv("CASHIER_CONF_PERSON", "0.50")),
        1: float(os.getenv("CASHIER_CONF_DRAWER", "0.45")),
        2: float(os.getenv("CASHIER_CONF_CASH",   "0.50")),
    }
    _CLASS_NAMES: Dict[int, str] = {0: "Person", 1: "Drawer_Open", 2: "Cash"}

    def __init__(self):
        model_path   = os.getenv("CASHIER_MODEL",        "./models/best_cashier.onnx")
        config_path  = os.getenv("CASHIER_CONFIG",       "./config/cashier_zones.yaml")
        evidence_dir = os.getenv("CASHIER_EVIDENCE_DIR", "./evidence/cashier")
        self._camera_id = os.getenv("CASHIER_CAMERA_ID", "cam")
        self.save       = os.getenv("SAVE_OUTPUT", "True").lower() in ("true", "1", "yes")

        log.info("[CASHIER] Loading model: %s", model_path)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"[CASHIER] Model not found: {model_path}")

        import onnxruntime as ort
        self._sess     = ort.InferenceSession(model_path)
        self._inp_name = self._sess.get_inputs()[0].name
        inp            = self._sess.get_inputs()[0].shape
        self._img_size = (int(inp[3]), int(inp[2]))

        self._config_path = config_path
        self._cfg         = _load_config(config_path)
        self._last_reload = time.monotonic()
        self._apply_config()

        self._cashier_poly, self._customer_poly = self._load_polys()

        self._frame_buffers  : Dict[str, deque]           = {}
        self._debounce_counts: Dict[str, Dict[str, int]]  = {}
        self._active_events  : Dict[str, Dict[str, Any]]  = {}

        self._frame_count: int = 0
        self._event_seq  : int = 0

        self._drawer_open_since  : Optional[float] = None
        self._customer_wait_since: Optional[float] = None

        self._evidence = _EvidenceWriter(evidence_dir, self._log_rotate_mb)

        self._drawer_totals_lock = threading.Lock()
        self._drawer_was_open: Dict[str, bool] = {}
        self._total_drawer_open_count: Dict[str, int] = {}
        self._total_drawer_open_duration_ms: Dict[str, int] = {}
        self._last_frame_monotonic: Dict[str, float] = {}
        self._drawer_totals_last_wall_save: float = 0.0
        try:
            self._drawer_duration_persist_interval = float(
                os.getenv("CASHIER_DRAWER_DURATION_PERSIST_SEC", "10")
            )
        except ValueError:
            self._drawer_duration_persist_interval = 10.0
        self._persist_drawer_totals = os.getenv(
            "CASHIER_DISABLE_DRAWER_TOTAL_PERSIST", ""
        ).lower() not in ("1", "true", "yes", "on")
        self._load_drawer_open_totals()

        log.info(
            "[CASHIER] Ready — model %s | img_size %s | classes %s | GIF %s",
            model_path, self._img_size, list(self._CLASS_NAMES.values()),
            "enabled" if _HAS_IMAGEIO else "disabled (imageio missing)",
        )

    # ─────────────────────────────────────────────
    # Config helpers
    # ─────────────────────────────────────────────

    def _apply_config(self) -> None:
        thr = self._cfg.get("thresholds", {})
        buf = self._cfg.get("buffer", {})
        gif = self._cfg.get("gif", {})
        deb = self._cfg.get("debounce", {})
        ev  = self._cfg.get("evidence", {})

        self._reload_interval = float(thr.get("config_reload_interval",
                                               os.getenv("CASHIER_RELOAD_INTERVAL", "60")))
        self._drawer_max      = float(thr.get("drawer_open_max_seconds",
                                               os.getenv("CASHIER_DRAWER_MAX_SEC", "30")))
        self._wait_max        = float(thr.get("customer_wait_max_seconds",
                                               os.getenv("CASHIER_WAIT_MAX_SEC", "30")))
        self._prox_iou        = float(thr.get("proximity_iou",
                                               os.getenv("CASHIER_PROXIMITY_IOU", "0.05")))

        self._buffer_size    = int(buf.get("size",         os.getenv("CASHIER_BUFFER_SIZE", "100")))
        self._jpeg_quality   = int(buf.get("jpeg_quality", "75"))

        self._gif_fps        = int(gif.get("fps",     "10"))
        self._gif_quality    = int(gif.get("quality", "85"))

        self._debounce_default = int(deb.get("default", _DEBOUNCE_DEFAULT))
        self._debounce_cfg     = {k: int(v) for k, v in deb.items() if k != "default"}

        self._save_gif       = bool(ev.get("save_gif",       True))
        self._save_thumbnail = bool(ev.get("save_thumbnail", True))
        self._log_rotate_mb  = float(ev.get("log_rotate_mb", 100.0))

        dt = thr.get("detection_threshold")
        if dt is None:
            dt = thr.get("detection_threshold_percent")
        if dt is not None:
            try:
                p = float(dt)
                self._global_conf_floor = max(0.0, min(1.0, p / 100.0))
            except (TypeError, ValueError):
                self._global_conf_floor = None
        else:
            self._global_conf_floor = None

        self._parse_detail_config(self._cfg.get("detail_config"))
        self._task_meta = self._cfg.get("task") if isinstance(self._cfg.get("task"), dict) else {}

    def _parse_detail_config(self, raw: Any) -> None:
        """Load CASHIER_BOX_OPEN detail_config (snake_case or camelCase)."""
        if not isinstance(raw, dict):
            raw = {}
        esl = raw.get("enable_staff_list", raw.get("enableStaffList", False))
        if isinstance(esl, str):
            esl = esl.lower() in ("true", "1", "yes", "on")
        self._enable_staff_list = bool(esl)
        ids = raw.get("staff_ids", raw.get("staffIds", []))
        if not isinstance(ids, (list, tuple)):
            ids = []
        self._staff_ids = set()
        for x in ids:
            try:
                self._staff_ids.add(int(x))
            except (TypeError, ValueError):
                continue
        dol = raw.get("drawer_open_limit", raw.get("drawerOpenLimit"))
        if dol is not None:
            try:
                self._drawer_max = float(dol)
            except (TypeError, ValueError):
                pass
        swl = raw.get("service_wait_limit", raw.get("serviceWaitLimit"))
        if swl is not None:
            try:
                self._wait_max = float(swl)
            except (TypeError, ValueError):
                pass

    def _min_conf(self, class_id: int) -> float:
        base = float(self._CONF.get(class_id, 0.5))
        if self._global_conf_floor is not None:
            return max(base, self._global_conf_floor)
        return base

    def _load_polys(self) -> Tuple[List[List[float]], List[List[float]]]:
        zones = self._cfg.get("zones", {})
        return (
            _build_poly(zones.get(ZONE_CASHIER,  {})),
            _build_poly(zones.get(ZONE_CUSTOMER, {})),
        )

    def _maybe_reload_config(self) -> None:
        if time.monotonic() - self._last_reload > self._reload_interval:
            self._cfg = _load_config(self._config_path)
            self._apply_config()
            self._cashier_poly, self._customer_poly = self._load_polys()
            self._last_reload = time.monotonic()
            log.debug("[CASHIER] Config reloaded from %s", self._config_path)

    def _roi_version(self) -> str:
        zones_json = json.dumps(self._cfg.get("zones", {}), sort_keys=True)
        return hashlib.md5(zones_json.encode()).hexdigest()[:8]

    # ─────────────────────────────────────────────
    # Zone assignment
    # ─────────────────────────────────────────────

    def _assign_zone(self, det, frame_w: int, frame_h: int) -> str:
        """Zone membership decided by bottom-centre foot point (normalised)."""
        foot_x = ((det.x1 + det.x2) / 2) / frame_w
        foot_y = det.y2 / frame_h
        if self._cashier_poly  and _point_in_poly(foot_x, foot_y, self._cashier_poly):
            return ZONE_CASHIER
        if self._customer_poly and _point_in_poly(foot_x, foot_y, self._customer_poly):
            return ZONE_CUSTOMER
        return ZONE_OUTSIDE

    def _match_binding_staff_id(
        self, person_bbox: List[float], bindings: List[Any]
    ) -> Optional[int]:
        best_iou = 0.0
        best_sid: Optional[int] = None
        for b in bindings:
            if not isinstance(b, dict):
                continue
            bb = b.get("bbox")
            if not bb or len(bb) < 4:
                continue
            bb_f = [float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])]
            iou_v = _iou(person_bbox, bb_f)
            if iou_v > best_iou and iou_v >= _STAFF_MATCH_IOU:
                best_iou = iou_v
                try:
                    best_sid = int(b["staff_id"])
                except (KeyError, TypeError, ValueError):
                    best_sid = None
        return best_sid

    def _resolve_cz_staff_state(
        self, cz: ZoneCount, context: Dict[str, Any]
    ) -> Tuple[bool, int]:
        """
        Returns (unauthorized_present, authorized_person_count_in_cz).
        Strict: enable_staff_list + empty bindings => all CZ persons unauthorized.
        """
        if not self._enable_staff_list:
            return False, cz.persons
        data = context.get("data") if isinstance(context.get("data"), dict) else {}
        staff_block = data.get(STAFF_CONTEXT_KEY) if isinstance(data, dict) else None
        if not isinstance(staff_block, dict):
            staff_block = {}
        bindings = staff_block.get("bindings")
        if not isinstance(bindings, list) or len(bindings) == 0:
            return (cz.persons > 0), 0
        auth = 0
        for pb in cz.person_bboxes:
            sid = self._match_binding_staff_id(pb, bindings)
            if sid is not None and sid in self._staff_ids:
                auth += 1
        return (cz.persons - auth) > 0, auth

    def _count_zones(self, detections, frame_w: int, frame_h: int) -> Tuple[ZoneCount, ZoneCount]:
        cz, kz = ZoneCount(), ZoneCount()
        for det in detections:
            cid = int(getattr(det, "class_id", -1))
            if float(getattr(det, "confidence", 0)) < self._min_conf(cid):
                continue
            zone = self._assign_zone(det, frame_w, frame_h)
            bbox = list(getattr(det, "bbox", []))
            if len(bbox) < 4:
                bbox = [float(det.x1), float(det.y1), float(det.x2), float(det.y2)]
            if zone == ZONE_CASHIER:
                if det.class_id == 0:
                    cz.persons += 1
                    cz.person_bboxes.append(bbox)
                    cz.person_confidences.append(round(det.confidence, 4))
                if det.class_id == 1: cz.drawers += 1; cz.drawer_bboxes.append(bbox)
                if det.class_id == 2: cz.cash    += 1; cz.cash_bboxes.append(bbox)
            elif zone == ZONE_CUSTOMER:
                if det.class_id == 0:
                    kz.persons += 1
                    kz.person_bboxes.append(bbox)
                    kz.person_confidences.append(round(det.confidence, 4))
                if det.class_id == 1: kz.drawers += 1; kz.drawer_bboxes.append(bbox)
                if det.class_id == 2: kz.cash    += 1; kz.cash_bboxes.append(bbox)
        return cz, kz

    # ─────────────────────────────────────────────
    # Person-level builder  (NEW — like PPE)
    # ─────────────────────────────────────────────

    def _build_persons(
        self,
        cz         : ZoneCount,
        kz         : ZoneCount,
        transaction: bool,
    ) -> List[PersonEntry]:
        """
        Build per-person entries (mirrors PPE schema).

        Linking strategy — drawer as proxy:
            person → drawer  (IoU >= prox_iou)
            drawer → cash    (IoU >= prox_iou)

        This handles the common case where cash sits inside an open drawer
        that is above the cashier's body bbox (no direct person↔cash overlap).
        """
        entries: List[PersonEntry] = []

        # ── Cashier-zone persons ──────────────────────────────────────
        for bbox, conf in zip(cz.person_bboxes, cz.person_confidences):
            entry = PersonEntry(
                person_bbox=bbox,
                confidence=conf,
                zone=ZONE_CASHIER,
            )

            # 1. Link drawers to this person
            linked_drawers: List[List[float]] = []
            for d_bbox in cz.drawer_bboxes:
                if _iou(bbox, d_bbox) >= self._prox_iou:
                    linked_drawers.append(d_bbox)
                    entry.items["drawers"].append({
                        "bbox"      : [int(c) for c in d_bbox],
                        "confidence": None,
                    })

            # 2. Link cash via drawer proxy (not directly from person)
            for d_bbox in linked_drawers:
                for c_bbox in cz.cash_bboxes:
                    if _iou(d_bbox, c_bbox) >= self._prox_iou:
                        entry.items["cash"].append({
                            "bbox"      : [int(c) for c in c_bbox],
                            "confidence": None,
                        })

            # 3. Mark transaction when person has both drawer + cash linked
            if transaction and entry.items["drawers"] and entry.items["cash"]:
                entry.transaction = True

            entries.append(entry)

        # ── Customer-zone persons ─────────────────────────────────────
        for bbox, conf in zip(kz.person_bboxes, kz.person_confidences):
            entry = PersonEntry(
                person_bbox=bbox,
                confidence=conf,
                zone=ZONE_CUSTOMER,
            )

            # Cash in customer zone linked directly to person
            for c_bbox in kz.cash_bboxes:
                if _iou(bbox, c_bbox) >= self._prox_iou:
                    entry.items["cash"].append({
                        "bbox"      : [int(c) for c in c_bbox],
                        "confidence": None,
                    })

            entries.append(entry)

        return entries

    # ─────────────────────────────────────────────
    # Business logic
    # ─────────────────────────────────────────────

    def _nearby(self, persons: List, drawers: List) -> bool:
        return any(_iou(p, d) >= self._prox_iou for p in persons for d in drawers)

    def _all_nearby(self, persons: List, drawers: List) -> bool:
        return all(
            any(_iou(p, d) >= self._prox_iou for d in drawers)
            for p in persons
        )

    def _enforce_single_open_drawer(self, detections: List[Any]) -> List[Any]:
        """
        Business rule:
        Only one drawer can be open at a time.
        If the model emits multiple drawer detections in a frame, keep the
        highest-confidence one and drop the rest.
        """
        drawer_dets = [d for d in detections if getattr(d, "class_id", None) == 1]
        if len(drawer_dets) <= 1:
            return detections

        best_drawer = max(drawer_dets, key=lambda d: float(getattr(d, "confidence", 0.0)))
        filtered = [
            d for d in detections
            if getattr(d, "class_id", None) != 1 or d is best_drawer
        ]
        log.debug(
            "[CASHIER] Multiple drawers detected (%d) — keeping top confidence %.3f",
            len(drawer_dets),
            float(getattr(best_drawer, "confidence", 0.0)),
        )
        return filtered

    def _evaluate(
        self,
        cz: ZoneCount,
        kz: ZoneCount,
        now: float,
        unauthorized_in_cz: bool,
    ) -> Tuple[str, str, List[str], bool, int, int]:
        """
        CASHIER_BOX_OPEN 14-rule first match (Section 8).
        Returns (case_id, severity, alerts, transaction, drawer_ms, wait_ms).
        """
        alerts: List[str] = []
        cash_any = (cz.cash + kz.cash) >= 1
        cash_zero = not cash_any
        drawer_open = cz.drawers >= 1

        if cz.drawers >= 1:
            if self._drawer_open_since is None:
                self._drawer_open_since = now
            drawer_elapsed = now - self._drawer_open_since
        else:
            self._drawer_open_since = None
            drawer_elapsed = 0.0

        if kz.persons >= 1 and cz.persons == 0:
            if self._customer_wait_since is None:
                self._customer_wait_since = now
            wait_elapsed = now - self._customer_wait_since
        else:
            self._customer_wait_since = None
            wait_elapsed = 0.0

        drawer_ms = int(drawer_elapsed * 1000) if drawer_open else 0
        wait_ms = int(wait_elapsed * 1000) if (kz.persons >= 1 and cz.persons == 0) else 0

        def ret(
            cid: str, sev: str, al: List[str], txn: bool
        ) -> Tuple[str, str, List[str], bool, int, int]:
            return cid, sev, al, txn, drawer_ms, wait_ms

        # Row 1 — A3
        if drawer_open and cash_any and cz.persons == 0:
            msg = "A3 CRITICAL: Cash + open drawer — register unguarded"
            alerts.append(msg)
            log.critical(
                "[CASHIER] %s | drawers=%d cash_cz=%d cash_kz=%d",
                msg, cz.drawers, cz.cash, kz.cash,
            )
            return ret("A3", SEVERITY_CRITICAL, alerts, False)

        # Row 2 — A4 (spec: non-staff + open drawer + cash), or legacy IoU when staff list off
        if drawer_open and cash_any:
            if self._enable_staff_list:
                if unauthorized_in_cz:
                    msg = "A4 CRITICAL: Unauthorised person at open register with cash"
                    alerts.append(msg)
                    log.critical(
                        "[CASHIER] %s | persons_cz=%d drawers=%d",
                        msg, cz.persons, cz.drawers,
                    )
                    return ret("A4", SEVERITY_CRITICAL, alerts, False)
            else:
                # Legacy: cash present but no person↔drawer proximity (staff may lack overlap)
                if (
                    cz.persons >= 1
                    and not self._nearby(cz.person_bboxes, cz.drawer_bboxes)
                ):
                    msg = "A4 CRITICAL: Unauthorised person at open register with cash"
                    alerts.append(msg)
                    log.critical(
                        "[CASHIER] %s | persons_cz=%d drawers=%d (IoU fallback)",
                        msg, cz.persons, cz.drawers,
                    )
                    return ret("A4", SEVERITY_CRITICAL, alerts, False)

        # Row 3 — A1 elevated (CRITICAL)
        if cz.persons == 0 and kz.persons >= 1 and drawer_open and cash_zero:
            msg = "A1 CRITICAL: Unattended open drawer — customer present"
            alerts.append(msg)
            log.warning("[CASHIER] %s | drawers=%d", msg, cz.drawers)
            return ret("A1", SEVERITY_CRITICAL, alerts, False)

        # Row 4 — A1 ALERT
        if cz.persons == 0 and drawer_open and cash_zero:
            msg = "A1 ALERT: Unattended open drawer"
            alerts.append(msg)
            log.warning("[CASHIER] %s | drawers=%d", msg, cz.drawers)
            return ret("A1", SEVERITY_ALERT, alerts, False)

        # Row 5 — A2
        if unauthorized_in_cz:
            msg = "A2 ALERT: Unexpected person in cashier zone"
            alerts.append(msg)
            log.warning("[CASHIER] %s | persons_cz=%d", msg, cz.persons)
            return ret("A2", SEVERITY_ALERT, alerts, False)

        # Row 6 — A7
        if cz.persons == 0 and not drawer_open and kz.cash >= 1:
            msg = "A7 ALERT: Cash in customer zone — no cashier present"
            alerts.append(msg)
            log.warning("[CASHIER] %s | cash_kz=%d", msg, kz.cash)
            return ret("A7", SEVERITY_ALERT, alerts, False)

        # Row 7 — A5
        if (
            cz.persons == 0
            and kz.persons >= 1
            and not drawer_open
            and cash_zero
            and wait_elapsed >= self._wait_max
        ):
            msg = f"A5 ALERT: Customer waiting {wait_elapsed:.0f}s — no cashier present"
            alerts.append(msg)
            log.warning("[CASHIER] %s", msg)
            return ret("A5", SEVERITY_ALERT, alerts, False)

        # Row 8 — A6 (authorized CZ staff only; unauthorized handled by A2)
        if (
            cz.persons >= 1
            and drawer_open
            and drawer_elapsed >= self._drawer_max
        ):
            msg = f"A6 ALERT: Drawer open {drawer_elapsed:.0f}s (limit {self._drawer_max:.0f}s)"
            alerts.append(msg)
            log.warning("[CASHIER] %s", msg)
            return ret("A6", SEVERITY_ALERT, alerts, False)

        # Row 9 — N4
        if (
            cz.persons >= 2
            and drawer_open
            and self._all_nearby(cz.person_bboxes, cz.drawer_bboxes)
        ):
            alerts.append("N4 EVENT: Staff handover / supervisor at register")
            log.info("[CASHIER] N4 | cashier_persons=%d", cz.persons)
            return ret("N4", SEVERITY_NORMAL, alerts, False)

        # Row 10 — N3
        if (
            cz.persons == 1
            and kz.persons >= 1
            and drawer_open
            and cash_any
            and self._nearby(cz.person_bboxes, cz.drawer_bboxes)
        ):
            alerts.append("N3 EVENT: Transaction in progress")
            log.info(
                "[CASHIER] N3 | cashier=%d drawer=%d customer=%d",
                cz.persons, cz.drawers, kz.persons,
            )
            return ret("N3", SEVERITY_NORMAL, alerts, True)

        # Row 11 — N6
        if (
            cz.persons == 1
            and drawer_open
            and cz.cash == 0
            and kz.cash == 0
        ):
            alerts.append("N6 EVENT: Drawer open, no cash (card / float check)")
            return ret("N6", SEVERITY_NORMAL, alerts, False)

        # Row 12 — N2
        if cz.persons >= 1 and not drawer_open:
            return ret("N2", SEVERITY_NORMAL, [], False)

        # Row 13 — N5
        if (
            cz.persons == 0
            and kz.persons >= 1
            and not drawer_open
            and cash_zero
            and wait_elapsed < self._wait_max
        ):
            return ret("N5", SEVERITY_NORMAL, [], False)

        # Row 14 — N1
        if cz.persons == 0 and kz.persons == 0 and not drawer_open and cash_zero:
            return ret("N1", SEVERITY_NORMAL, [], False)

        # Authorized CZ but no table row (e.g. 2+ staff, drawer open, not all IoU with drawer)
        if cz.persons >= 1 and not unauthorized_in_cz:
            msg = "A2 ALERT: Unexpected person in cashier zone"
            alerts.append(msg)
            log.warning("[CASHIER] %s | persons_cz=%d (fallback)", msg, cz.persons)
            return ret("A2", SEVERITY_ALERT, alerts, False)

        return ret("N1", SEVERITY_NORMAL, [], False)

    def _detections_for_person_structural(
        self, detections: List[Any], w: int, h: int, context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        data = context.get("data") if isinstance(context.get("data"), dict) else {}
        staff_block = data.get(STAFF_CONTEXT_KEY) if isinstance(data, dict) else None
        if not isinstance(staff_block, dict):
            staff_block = {}
        bindings = staff_block.get("bindings")
        if not isinstance(bindings, list):
            bindings = []
        out: List[Dict[str, Any]] = []
        for det in detections:
            cid = getattr(det, "class_id", None)
            if cid not in (0, 1, 2):
                continue
            conf = float(getattr(det, "confidence", 0))
            if conf < self._min_conf(int(cid)):
                continue
            zone = self._assign_zone(det, w, h)
            if zone == ZONE_OUTSIDE:
                continue
            zid = 1 if zone == ZONE_CASHIER else 2
            name = self._CLASS_NAMES.get(int(cid), str(cid))
            row: Dict[str, Any] = {
                "class": name,
                "confidence": round(conf, 4),
                "zone_id": zid,
            }
            if int(cid) == 0 and zone == ZONE_CASHIER and self._enable_staff_list:
                bbox = list(getattr(det, "bbox", []))
                if len(bbox) < 4:
                    bbox = [
                        float(det.x1), float(det.y1),
                        float(det.x2), float(det.y2),
                    ]
                sid = self._match_binding_staff_id(bbox, bindings)
                row["is_authorized"] = bool(sid is not None and sid in self._staff_ids)
            out.append(row)
        return out

    def _drawer_totals_path(self) -> Path:
        return self._evidence.base / "logs" / CASHIER_DRAWER_TOTALS_FILENAME

    def _load_drawer_open_totals(self) -> None:
        path = self._drawer_totals_path()
        if not path.is_file():
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            m = raw.get("by_camera")
            if isinstance(m, dict):
                out_c: Dict[str, int] = {}
                for k, v in m.items():
                    try:
                        iv = int(v)
                        if iv >= 0:
                            out_c[str(k)] = iv
                    except (TypeError, ValueError):
                        continue
                self._total_drawer_open_count = out_c
            dm = raw.get("total_open_duration_ms_by_camera")
            if isinstance(dm, dict):
                out_d: Dict[str, int] = {}
                for k, v in dm.items():
                    try:
                        iv = int(v)
                        if iv >= 0:
                            out_d[str(k)] = iv
                    except (TypeError, ValueError):
                        continue
                self._total_drawer_open_duration_ms = out_d
            log.info(
                "[CASHIER] Loaded drawer totals: %d camera(s) count, %d duration map",
                len(self._total_drawer_open_count),
                len(self._total_drawer_open_duration_ms),
            )
        except Exception as exc:
            log.warning("[CASHIER] Drawer totals load failed (%s) — starting empty", exc)
            self._total_drawer_open_count = {}
            self._total_drawer_open_duration_ms = {}

    def _save_drawer_open_totals_nolock(self) -> None:
        if not self._persist_drawer_totals:
            return
        path = self._drawer_totals_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "by_camera": dict(self._total_drawer_open_count),
                "total_open_duration_ms_by_camera": dict(self._total_drawer_open_duration_ms),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp.replace(path)
            self._drawer_totals_last_wall_save = time.time()
        except Exception as exc:
            log.warning("[CASHIER] Drawer totals save failed: %s", exc)

    def _update_drawer_open_metrics(
        self, camera_id: str, drawer_open: bool, now_mono: float
    ) -> Tuple[int, int]:
        """
        Cumulative open count (rising edge) + cumulative open duration (ms).
        Duration: between consecutive frames, time is added if the *previous* frame
        had the cashier drawer open (matches discrete sampling of open intervals).
        Persists on open edge, close edge, and periodically while open.
        """
        with self._drawer_totals_lock:
            prev = self._drawer_was_open.get(camera_id, False)
            last_t = self._last_frame_monotonic.get(camera_id)
            if last_t is not None:
                dt = max(0.0, now_mono - last_t)
                if prev:
                    self._total_drawer_open_duration_ms[camera_id] = int(
                        self._total_drawer_open_duration_ms.get(camera_id, 0)
                    ) + int(dt * 1000)
            self._last_frame_monotonic[camera_id] = now_mono

            cur_c = int(self._total_drawer_open_count.get(camera_id, 0))
            if drawer_open and not prev:
                cur_c += 1
                self._total_drawer_open_count[camera_id] = cur_c

            self._drawer_was_open[camera_id] = drawer_open

            cur_d = int(self._total_drawer_open_duration_ms.get(camera_id, 0))

            edge_dirty = (drawer_open and not prev) or (prev and not drawer_open)
            throttle_dirty = False
            if (
                not edge_dirty
                and prev
                and drawer_open
                and self._drawer_duration_persist_interval > 0
            ):
                wall = time.time()
                if wall - self._drawer_totals_last_wall_save >= self._drawer_duration_persist_interval:
                    throttle_dirty = True

            if edge_dirty or throttle_dirty:
                self._save_drawer_open_totals_nolock()

            return cur_c, cur_d

    def _build_person_structural(
        self,
        case_id: str,
        severity: str,
        cz: ZoneCount,
        kz: ZoneCount,
        unauthorized: bool,
        detections_ps: List[Dict[str, Any]],
        drawer_ms: int,
        wait_ms: int,
        total_open_count: int,
        total_open_duration_ms: int,
    ) -> str:
        alert = severity in (SEVERITY_ALERT, SEVERITY_CRITICAL)
        crit = severity == SEVERITY_CRITICAL
        body: Dict[str, Any] = {
            "case_matched": case_id,
            "case_level": severity,
            "alert_triggered": alert,
            "critical_triggered": crit,
            "total_open_count": int(total_open_count),
            "total_open_duration_ms": int(total_open_duration_ms),
            "current_open_duration_ms": int(drawer_ms),
            "zones": {
                "cashier": {
                    "persons_count": cz.persons,
                    "drawers_count": cz.drawers,
                    "cash_count": cz.cash,
                    "unauthorized_present": unauthorized,
                },
                "customer": {
                    "persons_count": kz.persons,
                    "cash_count": kz.cash,
                },
            },
            "detections": detections_ps,
        }
        if case_id in ("A1", "A3", "A4", "A6") and drawer_ms > 0:
            body["drawer_open_duration_ms"] = drawer_ms
        if case_id == "A5" and wait_ms > 0:
            body["wait_duration_ms"] = wait_ms
        return json.dumps(body, separators=(",", ":"))

    # ─────────────────────────────────────────────
    # Frame buffer helpers
    # ─────────────────────────────────────────────

    def _push_frame(self, camera_id: str, frame: np.ndarray) -> None:
        buf = self._frame_buffers.setdefault(
            camera_id, deque(maxlen=self._buffer_size)
        )
        _, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality])
        buf.append(encoded.tobytes())

    def _get_pre_frames(self, camera_id: str, n: int) -> List[bytes]:
        buf = self._frame_buffers.get(camera_id, deque())
        frames = list(buf)
        return frames[-n:] if n > 0 and frames else []

    # ─────────────────────────────────────────────
    # Debounce
    # ─────────────────────────────────────────────

    def _check_debounce(self, camera_id: str, case_id: str) -> bool:
        cam_counts = self._debounce_counts.setdefault(camera_id, {})
        for k in list(cam_counts.keys()):
            if k != case_id:
                cam_counts[k] = 0
        cam_counts[case_id] = cam_counts.get(case_id, 0) + 1
        threshold = self._debounce_cfg.get(case_id, self._debounce_default)
        return cam_counts[case_id] >= threshold

    # ─────────────────────────────────────────────
    # Event lifecycle
    # ─────────────────────────────────────────────

    def _build_meta(
        self,
        camera_id: str,
        case_id  : str,
        severity : str,
        alerts   : List[str],
        cz       : ZoneCount,
        kz       : ZoneCount,
        frame_id : int,
    ) -> Dict:
        self._event_seq += 1
        ts = datetime.now(timezone.utc)
        return {
            "event_id"   : f"{case_id}_{ts.strftime('%Y%m%d_%H%M%S')}_{self._event_seq:03d}",
            "version"    : self._cfg.get("meta", {}).get("version", "1.2.0"),
            "type"       : case_id,
            "level"      : severity,
            "reason"     : alerts[0] if alerts else case_id,
            "timestamp"  : ts.isoformat(),
            "frame_id"   : frame_id,
            "camera_id"  : camera_id,
            "roi_version": self._roi_version(),
            "zones"      : {"cashier": cz.to_dict(), "customer": kz.to_dict()},
            "gif_path"   : None,
            "thumb_path" : None,
            "duration_s" : None,
            "resolved_at": None,
        }

    def _start_event(
        self,
        camera_id: str,
        case_id  : str,
        meta     : Dict,
        frame    : np.ndarray,
        now      : float,
    ) -> None:
        budget     = _GIF_BUDGET.get(case_id, {})
        pre_n      = budget.get("pre", 0)
        pre_frames = self._get_pre_frames(camera_id, pre_n)

        _, enc = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality])
        post_frames = [enc.tobytes()]

        self._active_events[camera_id] = {
            "case_id"         : case_id,
            "start_time"      : now,
            "pre_frames"      : pre_frames,
            "post_frames"     : post_frames,
            "last_keyframe_t" : now,
            "meta"            : meta,
            "_long_warned"    : False,
        }
        self._evidence.append_log({**meta, "status": "triggered"})
        log.debug("[CASHIER] Event started: %s cam=%s", case_id, camera_id)

    def _accumulate_post(
        self, camera_id: str, case_id: str, frame: np.ndarray, now: float
    ) -> None:
        ev = self._active_events.get(camera_id)
        if not ev:
            return

        budget   = _GIF_BUDGET.get(case_id, {})
        max_post = budget.get("max_post", 600)
        kf_int   = budget.get("keyframe_interval")

        elapsed = now - float(ev["start_time"])
        if max_post is None and elapsed >= _CRITICAL_UNBOUND_WARN_SEC and not ev.get("_long_warned"):
            log.warning(
                "[CASHIER] Case %s event > %.0fs — unbounded post buffer (memory risk on edge)",
                case_id,
                _CRITICAL_UNBOUND_WARN_SEC,
            )
            ev["_long_warned"] = True

        if max_post is not None and len(ev["post_frames"]) >= max_post:
            return

        if kf_int is not None:
            if now - ev["last_keyframe_t"] < kf_int:
                return
            ev["last_keyframe_t"] = now

        _, enc = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality])
        ev["post_frames"].append(enc.tobytes())

    def _check_resolved(self, camera_id: str, new_case_id: str, now: float) -> bool:
        ev = self._active_events.get(camera_id)
        if not ev:
            return False
        if new_case_id == ev["case_id"]:
            return False

        ev["duration_s"]  = now - ev["start_time"]
        ev["resolved_at"] = datetime.now(timezone.utc).isoformat()

        if self._save_gif:
            self._finalize_gif_async(camera_id, dict(ev))

        self._active_events.pop(camera_id, None)
        log.debug("[CASHIER] Event resolved: %s → %s (%.1fs)", ev["case_id"], new_case_id, ev["duration_s"])
        return True

    def _finalize_gif_async(self, camera_id: str, event: Dict) -> None:
        threading.Thread(
            target=self._compile_gif,
            args=(camera_id, event),
            daemon=True,
        ).start()

    def _compile_gif(self, camera_id: str, event: Dict) -> None:
        case_id    = event["case_id"]
        budget     = _GIF_BUDGET.get(case_id, {})
        fps        = budget.get("fps", self._gif_fps)
        all_frames = event.get("pre_frames", []) + event.get("post_frames", [])

        gif_path, thumb_path = self._evidence.compile_gif(
            frames=all_frames,
            case_id=case_id,
            camera_id=camera_id,
            fps=fps,
            quality=self._gif_quality,
        )

        meta = event.get("meta", {})
        self._evidence.append_log({
            **meta,
            "status"     : "resolved",
            "duration_s" : event.get("duration_s"),
            "resolved_at": event.get("resolved_at"),
            "gif_path"   : gif_path,
            "thumb_path" : thumb_path,
        })

    # ─────────────────────────────────────────────
    # Annotation
    # ─────────────────────────────────────────────

    def _draw_zone_badges(
        self, out: np.ndarray, cz: ZoneCount, kz: ZoneCount
    ) -> None:
        for bbox_list, color, label in (
            (cz.person_bboxes + cz.drawer_bboxes + cz.cash_bboxes, _CLR_CASHIER,  "[CZ]"),
            (kz.person_bboxes + kz.drawer_bboxes + kz.cash_bboxes, _CLR_CUSTOMER, "[KZ]"),
        ):
            for bbox in bbox_list:
                x1, y1 = int(bbox[0]), int(bbox[1])
                bg_y1  = max(y1 - 18, 0)
                bg_y2  = max(y1 - 1, 17)
                x2_bg  = x1 + 36
                cv2.rectangle(out, (x1, bg_y1), (x2_bg, bg_y2), color, -1)
                cv2.putText(
                    out, label, (x1 + 2, bg_y2 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA,
                )

    def _draw(self, frame: np.ndarray, result: CashierResult) -> np.ndarray:
        out = frame.copy()
        h, w = out.shape[:2]

        severity_color = {
            SEVERITY_NORMAL  : _CLR_NORMAL,
            SEVERITY_ALERT   : _CLR_ALERT,
            SEVERITY_CRITICAL: _CLR_CRITICAL,
        }.get(result.severity, _CLR_NORMAL)

        for poly_norm, color, label in (
            (self._cashier_poly,  _CLR_CASHIER,  ZONE_CASHIER),
            (self._customer_poly, _CLR_CUSTOMER, ZONE_CUSTOMER),
        ):
            if not poly_norm:
                continue
            pts = np.array(_denorm_poly(poly_norm, w, h), dtype=np.int32)
            overlay = out.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.12, out, 0.88, 0, out)
            cv2.polylines(out, [pts], isClosed=True, color=color, thickness=2)
            cv2.putText(out, label, (pts[0][0] + 4, pts[0][1] + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)

        self._draw_zone_badges(out, result.cashier_zone, result.customer_zone)

        for i, alert in enumerate(result.alerts):
            cv2.putText(out, alert, (10, 28 + i * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, severity_color, 2, cv2.LINE_AA)

        badge = f"{result.case_id} | {result.severity}"
        (bw, bh), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
        cv2.rectangle(out, (w - bw - 14, h - bh - 16), (w - 2, h - 2), severity_color, -1)
        cv2.putText(out, badge, (w - bw - 10, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 2, cv2.LINE_AA)

        for i, (label, zc) in enumerate((
            ("CZ", result.cashier_zone),
            ("KZ", result.customer_zone),
        )):
            line = f"{label}: P={zc.persons} D={zc.drawers} C={zc.cash}"
            cv2.putText(out, line, (8, h - 10 - (1 - i) * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        fc_text = f"#{result.frame_id}"
        (fw, _), _ = cv2.getTextSize(fc_text, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
        cv2.putText(out, fc_text, (w - fw - 6, h - bh - 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA)

        return out

    # ─────────────────────────────────────────────
    # Service callable
    # ─────────────────────────────────────────────

    def __call__(self, context: Dict[str, Any]) -> Dict[str, Any]:
        self._maybe_reload_config()

        frame      = context["data"]["frame"]
        detections = context["data"]["detection"].get("items", [])
        h, w       = frame.shape[:2]
        now        = time.monotonic()
        camera_id  = context.get("camera_id") or self._camera_id

        # 1. Push raw frame to per-camera buffer (before annotation)
        self._push_frame(camera_id, frame)
        self._frame_count += 1
        frame_id = self._frame_count

        # 2. Zone assignment + counts (uses full detections incl drawers/cash)
        detections = self._enforce_single_open_drawer(detections)
        cz, kz = self._count_zones(detections, w, h)

        unauthorized_in_cz, _auth_cz = self._resolve_cz_staff_state(cz, context)

        drawer_open_now = cz.drawers >= 1
        total_open_count, total_open_duration_ms = self._update_drawer_open_metrics(
            camera_id, drawer_open_now, now
        )

        # 3. Business logic evaluation (14-rule table)
        (
            case_id,
            severity,
            alerts,
            transaction,
            drawer_ms,
            wait_ms,
        ) = self._evaluate(cz, kz, now, unauthorized_in_cz)

        if os.getenv("CASHIER_DEBUG_ZONES", "").lower() in ("1", "true", "yes", "on"):
            log.info(
                "[CASHIER] zones | frame=%d | CZ P=%d D=%d C=%d | KZ P=%d D=%d C=%d | "
                "case=%s | transaction=%s | alerts=%s",
                frame_id,
                cz.persons,
                cz.drawers,
                cz.cash,
                kz.persons,
                kz.drawers,
                kz.cash,
                case_id,
                transaction,
                alerts if alerts else "—",
            )

        # 4. Detect event resolution (alert case changed)
        self._check_resolved(camera_id, case_id, now)

        # 5. Debounce gate
        debounced = self._check_debounce(camera_id, case_id)

        # 6. Build person-level entries (NEW — like PPE)
        #    Must run after _count_zones and _evaluate (needs transaction flag)
        persons = self._build_persons(cz, kz, transaction)

        dets_ps = self._detections_for_person_structural(detections, w, h, context)
        ps_str = self._build_person_structural(
            case_id,
            severity,
            cz,
            kz,
            unauthorized_in_cz,
            dets_ps,
            drawer_ms,
            wait_ms,
            total_open_count,
            total_open_duration_ms,
        )

        d_drawer = drawer_ms if case_id in ("A1", "A3", "A4", "A6") else None
        d_wait = wait_ms if case_id == "A5" else None

        # 7. Build result
        result = CashierResult(
            cashier_zone           = cz,
            customer_zone          = kz,
            case_id                = case_id,
            severity               = severity,
            alerts                 = alerts,
            transaction            = transaction,
            frame_saved            = False,
            evidence_path          = None,
            persons                = persons,
            frame_id               = frame_id,
            person_structural      = ps_str,
            unauthorized_present   = unauthorized_in_cz,
            drawer_open_duration_ms= d_drawer,
            wait_duration_ms       = d_wait,
            task_meta                = dict(self._task_meta) if self._task_meta else None,
            total_open_count         = total_open_count,
            total_open_duration_ms   = total_open_duration_ms,
            current_open_duration_ms = drawer_ms,
        )

        # 8. Annotate live frame
        #    _draw uses cz/kz directly — not affected by detection filter below
        if self.save:
            context["data"]["frame"] = self._draw(frame, result)

        annotated = context["data"]["frame"] if self.save else frame

        # 9. Evidence logic
        should_save = severity in (SEVERITY_ALERT, SEVERITY_CRITICAL)

        if should_save and debounced:
            if camera_id not in self._active_events:
                meta = self._build_meta(camera_id, case_id, severity, alerts, cz, kz, frame_id)
                img_path = self._evidence.save_frame(annotated, case_id, meta, camera_id, self._gif_quality)
                result.frame_saved   = True
                result.evidence_path = img_path
                meta["thumb_path"]   = img_path
                self._start_event(camera_id, case_id, meta, annotated, now)
            else:
                self._accumulate_post(camera_id, case_id, annotated, now)

        elif case_id == "N3" and transaction:
            meta     = self._build_meta(camera_id, case_id, severity, alerts, cz, kz, frame_id)
            img_path = self._evidence.save_frame(annotated, case_id, meta, camera_id, self._gif_quality)
            self._evidence.append_log({**meta, "status": "triggered"})
            result.frame_saved   = True
            result.evidence_path = img_path

        elif camera_id in self._active_events:
            self._accumulate_post(camera_id, case_id, annotated, now)

        # 10. Publish to API state tracker
        try:
            from apis.cashier import push_result
            push_result(camera_id, result.to_dict())
        except Exception:
            pass

        # 11. Write cashier output to context (Hybrid schema)
        context["data"]["use_case"]["cashier"] = result.to_dict()

        # 12. Filter detection output — keep persons only (class_id == 0)
        #     Drawers and cash are now inside cashier.persons[].items
        #     Must run LAST so all cashier logic above sees full detections
        context["data"]["detection"]["items"] = [
            d for d in detections if getattr(d, "class_id", None) == 0
        ]
        context["data"]["detection"]["count"] = len(
            context["data"]["detection"]["items"]
        )

        return context


# ─────────────────────────────────────────────
# FrameBus task — CASHIER_DRAWER
# ─────────────────────────────────────────────
#
# FrameBus fans the same YOLO track output to every task on a channel. For
# drawer/cash classes in payload["detection"]["items"], set YOLO_MODEL (etc.)
# to cashier weights — avoid mixing COCO-only tasks on the same channel unless
# class indices align (0=person, 1=drawer, 2=cash).

class CashierDrawerTask:
    """Register via POST /api/tasks with algorithmType CASHIER_DRAWER."""

    def __init__(self, task_config: dict):
        self._task_config = task_config
        self._svc = CashierService()

    def __call__(self, payload: Dict[str, Any]) -> List[Any]:
        if not self._task_config.get("enable", True):
            return []

        camera_id = str(payload.get("camera_id", ""))
        detection = payload.get("detection") or {}
        items = detection.get("items", [])

        context: Dict[str, Any] = {
            "camera_id": camera_id,
            "data": {
                "frame": payload["frame"],
                "detection": {"items": list(items), "count": len(items)},
                "use_case": {},
            },
        }
        self._svc(context)
        return []