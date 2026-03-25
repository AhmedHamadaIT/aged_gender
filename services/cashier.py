"""
services/cashier.py  —  v1.1.0

CashierService — ROI-based cashier drawer monitor.

Model   : YOLO11m  (best_cashier.onnx)
Classes : 0=Person  1=Drawer_Open  2=Cash
Zones   : ROI_CASHIER (staff side)  •  ROI_CUSTOMER (customer side)

Reads   context["data"]["detection"]["items"]   — List[Detection]
        context["data"]["frame"]                — np.ndarray BGR

Writes  context["data"]["use_case"]["cashier"]  — dict
        {
          "cashier_zone"  : { "persons": int, "drawers": int, "cash": int },
          "customer_zone" : { "persons": int, "drawers": int, "cash": int },
          "case_id"       : str,       # N1–N6 / A1–A7
          "severity"      : str,       # NORMAL | ALERT | CRITICAL
          "alerts"        : List[str],
          "transaction"   : bool,
          "frame_saved"   : bool,
          "evidence_path" : str | None,
          "frame_id"      : int,
          "timestamp"     : str,
        }

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
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
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
_GIF_BUDGET: Dict[str, Dict] = {
    "A1": {"pre": 30,  "max_post": 600,  "fps": 10, "keyframe_interval": None},
    "A2": {"pre": 50,  "max_post": 300,  "fps": 10, "keyframe_interval": None},
    "A3": {"pre": 60,  "max_post": 5000, "fps": 10, "keyframe_interval": None},
    "A4": {"pre": 60,  "max_post": 5000, "fps": 10, "keyframe_interval": None},
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
        "meta": {"version": "1.0.0"},
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
    }


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
    person_confidences : List[float]       = field(default_factory=list)  # confidence per person

    def to_dict(self) -> Dict:
        return {"persons": self.persons, "drawers": self.drawers, "cash": self.cash}


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
    frame_id      : int = 0
    timestamp     : str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict:
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
        return {
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
        frames   : List[bytes],   # JPEG-encoded bytes
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

        # Save thumbnail from first frame
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
    """

    # Per-class confidence thresholds (overridable via env)
    _CONF: Dict[int, float] = {
        0: float(os.getenv("CASHIER_CONF_PERSON", "0.50")),  # Person
        1: float(os.getenv("CASHIER_CONF_DRAWER", "0.45")),  # Drawer_Open
        2: float(os.getenv("CASHIER_CONF_CASH",   "0.50")),  # Cash
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
        self._img_size = (int(inp[3]), int(inp[2]))   # (W, H)

        # Config
        self._config_path = config_path
        self._cfg         = _load_config(config_path)
        self._last_reload = time.monotonic()
        self._apply_config()

        self._cashier_poly, self._customer_poly = self._load_polys()

        # Per-camera rolling frame buffers (JPEG-encoded bytes)
        self._frame_buffers  : Dict[str, deque]           = {}
        # Per-camera debounce counters  {cam_id: {case_id: consecutive_count}}
        self._debounce_counts: Dict[str, Dict[str, int]]  = {}
        # Per-camera active event trackers
        self._active_events  : Dict[str, Dict[str, Any]]  = {}

        # Global frame counter + event sequence
        self._frame_count: int = 0
        self._event_seq  : int = 0

        # Business logic timers
        self._drawer_open_since  : Optional[float] = None
        self._customer_wait_since: Optional[float] = None

        # Evidence writer
        self._evidence = _EvidenceWriter(evidence_dir, self._log_rotate_mb)

        log.info(
            "[CASHIER] Ready — model %s | img_size %s | classes %s | GIF %s",
            model_path, self._img_size, list(self._CLASS_NAMES.values()),
            "enabled" if _HAS_IMAGEIO else "disabled (imageio missing)",
        )

    # ─────────────────────────────────────────────
    # Config helpers
    # ─────────────────────────────────────────────

    def _apply_config(self) -> None:
        """Read all config sections and apply to instance attributes."""
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
        """MD5 hash (first 8 chars) of current zone config — for metadata tracing."""
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

    def _count_zones(self, detections, frame_w: int, frame_h: int) -> Tuple[ZoneCount, ZoneCount]:
        cz, kz = ZoneCount(), ZoneCount()
        for det in detections:
            if det.confidence < self._CONF.get(det.class_id, 0.5):
                continue
            zone = self._assign_zone(det, frame_w, frame_h)
            bbox = list(det.bbox)
            if zone == ZONE_CASHIER:
                if det.class_id == 0:
                    cz.persons += 1
                    cz.person_bboxes.append(bbox)
                    cz.person_confidences.append(round(det.confidence, 4))
                if det.class_id == 1: cz.drawers  += 1; cz.drawer_bboxes.append(bbox)
                if det.class_id == 2: cz.cash     += 1; cz.cash_bboxes.append(bbox)
            elif zone == ZONE_CUSTOMER:
                if det.class_id == 0:
                    kz.persons += 1
                    kz.person_bboxes.append(bbox)
                    kz.person_confidences.append(round(det.confidence, 4))
                if det.class_id == 1: kz.drawers  += 1; kz.drawer_bboxes.append(bbox)
                if det.class_id == 2: kz.cash     += 1; kz.cash_bboxes.append(bbox)
        return cz, kz

    # ─────────────────────────────────────────────
    # Business logic
    # ─────────────────────────────────────────────

    def _nearby(self, persons: List, drawers: List) -> bool:
        """True if any person/drawer pair has IoU ≥ threshold."""
        return any(_iou(p, d) >= self._prox_iou for p in persons for d in drawers)

    def _all_nearby(self, persons: List, drawers: List) -> bool:
        """True if EVERY person in the list is near at least one drawer (N4 check)."""
        return all(
            any(_iou(p, d) >= self._prox_iou for d in drawers)
            for p in persons
        )

    def _evaluate(
        self, cz: ZoneCount, kz: ZoneCount, now: float
    ) -> Tuple[str, str, List[str], bool]:
        """
        Evaluate business logic rules (first match wins) and return:
            (case_id, severity, alerts, is_transaction)
        """
        alerts: List[str] = []

        # ── Timers ──
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

        # ── CRITICAL ──

        # A3 — Cash + open drawer, no cashier (theft signature)
        if cz.drawers >= 1 and (cz.cash >= 1 or kz.cash >= 1) and cz.persons == 0:
            msg = "A3 CRITICAL: Cash + open drawer — register unguarded"
            alerts.append(msg)
            log.critical("[CASHIER] %s | drawers=%d cash_cz=%d cash_kz=%d", msg, cz.drawers, cz.cash, kz.cash)
            return "A3", SEVERITY_CRITICAL, alerts, False

        # A4 — Non-staff in cashier zone with open drawer + cash (person NOT proximate → intruder)
        if cz.persons >= 1 and cz.drawers >= 1 and (cz.cash >= 1 or kz.cash >= 1):
            if not self._nearby(cz.person_bboxes, cz.drawer_bboxes):
                msg = "A4 CRITICAL: Unauthorised person at open register with cash"
                alerts.append(msg)
                log.critical("[CASHIER] %s | persons_cz=%d drawers=%d", msg, cz.persons, cz.drawers)
                return "A4", SEVERITY_CRITICAL, alerts, False

        # ── ALERT ──

        # A1 — Unattended open drawer (elevated to CRITICAL if customer present)
        if cz.drawers >= 1 and cz.persons == 0:
            extra    = " — customer present" if kz.persons >= 1 else ""
            severity = SEVERITY_CRITICAL if kz.persons >= 1 else SEVERITY_ALERT
            level    = "CRITICAL" if severity == SEVERITY_CRITICAL else "ALERT"
            msg = f"A1 {level}: Unattended open drawer{extra}"
            alerts.append(msg)
            log.warning("[CASHIER] %s | drawers=%d customer_persons=%d", msg, cz.drawers, kz.persons)
            return "A1", severity, alerts, False

        # A5 — Customer waiting too long without cashier
        if wait_elapsed >= self._wait_max:
            msg = f"A5 ALERT: Customer waiting {wait_elapsed:.0f}s — no cashier present"
            alerts.append(msg)
            log.warning("[CASHIER] %s", msg)
            return "A5", SEVERITY_ALERT, alerts, False

        # A6 — Drawer open too long
        if drawer_elapsed >= self._drawer_max:
            msg = f"A6 ALERT: Drawer open {drawer_elapsed:.0f}s (limit {self._drawer_max:.0f}s)"
            alerts.append(msg)
            log.warning("[CASHIER] %s", msg)
            return "A6", SEVERITY_ALERT, alerts, False

        # A7 — Cash in customer zone, no cashier
        if kz.cash >= 1 and cz.persons == 0 and cz.drawers == 0:
            msg = "A7 ALERT: Cash in customer zone — no cashier present"
            alerts.append(msg)
            log.warning("[CASHIER] %s | cash_kz=%d", msg, kz.cash)
            return "A7", SEVERITY_ALERT, alerts, False

        # ── NORMAL ──

        # N3 — Active transaction (cashier + drawer + customer + cash, nearby)
        if (cz.persons >= 1 and cz.drawers >= 1 and kz.persons >= 1
                and (cz.cash >= 1 or kz.cash >= 1)
                and self._nearby(cz.person_bboxes, cz.drawer_bboxes)):
            alerts.append("N3 EVENT: Transaction in progress")
            log.info("[CASHIER] N3 | cashier=%d drawer=%d customer=%d", cz.persons, cz.drawers, kz.persons)
            return "N3", SEVERITY_NORMAL, alerts, True

        # N4 — Two staff at register (handover / supervisor) — ALL persons must be near a drawer
        if (cz.persons >= 2 and cz.drawers >= 1
                and self._all_nearby(cz.person_bboxes, cz.drawer_bboxes)):
            alerts.append("N4 EVENT: Staff handover / supervisor at register")
            log.info("[CASHIER] N4 | cashier_persons=%d", cz.persons)
            return "N4", SEVERITY_NORMAL, alerts, False

        # N6 — Drawer open, no cash (card transaction / float check)
        if cz.persons >= 1 and cz.drawers >= 1 and cz.cash == 0 and kz.cash == 0:
            alerts.append("N6 EVENT: Drawer open, no cash (card / float check)")
            return "N6", SEVERITY_NORMAL, alerts, False

        # N5 — Customer waiting (no transaction started yet)
        if kz.persons >= 1 and cz.drawers == 0:
            return "N5", SEVERITY_NORMAL, [], False

        # N2 — Cashier on duty, no customer
        if cz.persons >= 1 and cz.drawers == 0 and kz.persons == 0:
            return "N2", SEVERITY_NORMAL, [], False

        # A2 — Unexpected person in cashier zone (any person, no other rule matched)
        if cz.persons >= 1:
            msg = "A2 ALERT: Unexpected person in cashier zone"
            alerts.append(msg)
            log.warning("[CASHIER] %s | persons_cz=%d", msg, cz.persons)
            return "A2", SEVERITY_ALERT, alerts, False

        # N1 — Idle register
        return "N1", SEVERITY_NORMAL, [], False

    # ─────────────────────────────────────────────
    # Frame buffer helpers
    # ─────────────────────────────────────────────

    def _push_frame(self, camera_id: str, frame: np.ndarray) -> None:
        """Encode frame as JPEG and push to the per-camera rolling buffer."""
        buf = self._frame_buffers.setdefault(
            camera_id, deque(maxlen=self._buffer_size)
        )
        _, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality])
        buf.append(encoded.tobytes())

    def _get_pre_frames(self, camera_id: str, n: int) -> List[bytes]:
        """Return the last `n` frames from the buffer (oldest first)."""
        buf = self._frame_buffers.get(camera_id, deque())
        frames = list(buf)
        return frames[-n:] if n > 0 and frames else []

    # ─────────────────────────────────────────────
    # Debounce
    # ─────────────────────────────────────────────

    def _check_debounce(self, camera_id: str, case_id: str) -> bool:
        """
        Increment consecutive counter for case_id.
        Reset counters for other cases.
        Returns True when the threshold is reached.
        """
        cam_counts = self._debounce_counts.setdefault(camera_id, {})

        # Reset all other case counters
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
            "version"    : self._cfg.get("meta", {}).get("version", "1.0.0"),
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
        """Open a new active event for the given camera."""
        budget     = _GIF_BUDGET.get(case_id, {})
        pre_n      = budget.get("pre", 0)
        pre_frames = self._get_pre_frames(camera_id, pre_n)

        _, enc = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality])
        post_frames = [enc.tobytes()]

        self._active_events[camera_id] = {
            "case_id"        : case_id,
            "start_time"     : now,
            "pre_frames"     : pre_frames,
            "post_frames"    : post_frames,
            "last_keyframe_t": now,
            "meta"           : meta,
        }
        # Append trigger record to JSONL
        self._evidence.append_log({**meta, "status": "triggered"})
        log.debug("[CASHIER] Event started: %s cam=%s", case_id, camera_id)

    def _accumulate_post(
        self, camera_id: str, case_id: str, frame: np.ndarray, now: float
    ) -> None:
        """Add a post-event frame (or keyframe) to the active event."""
        ev     = self._active_events.get(camera_id)
        if not ev:
            return

        budget   = _GIF_BUDGET.get(case_id, {})
        max_post = budget.get("max_post", 600)
        kf_int   = budget.get("keyframe_interval")  # None = every frame

        if len(ev["post_frames"]) >= max_post:
            return  # cap reached

        # Keyframe sampling for A5/A6
        if kf_int is not None:
            if now - ev["last_keyframe_t"] < kf_int:
                return
            ev["last_keyframe_t"] = now

        _, enc = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality])
        ev["post_frames"].append(enc.tobytes())

    def _check_resolved(self, camera_id: str, new_case_id: str, now: float) -> bool:
        """Detect when an active alert event transitions to a different case."""
        ev = self._active_events.get(camera_id)
        if not ev:
            return False
        if new_case_id == ev["case_id"]:
            return False

        # Event resolved
        ev["duration_s"]  = now - ev["start_time"]
        ev["resolved_at"] = datetime.now(timezone.utc).isoformat()

        if self._save_gif:
            self._finalize_gif_async(camera_id, dict(ev))

        self._active_events.pop(camera_id, None)
        log.debug("[CASHIER] Event resolved: %s → %s (%.1fs)", ev["case_id"], new_case_id, ev["duration_s"])
        return True

    def _finalize_gif_async(self, camera_id: str, event: Dict) -> None:
        """Compile GIF in a background daemon thread — never blocks the pipeline."""
        threading.Thread(
            target=self._compile_gif,
            args=(camera_id, event),
            daemon=True,
        ).start()

    def _compile_gif(self, camera_id: str, event: Dict) -> None:
        """Background thread: compile GIF, save thumbnail, update JSONL log."""
        case_id  = event["case_id"]
        budget   = _GIF_BUDGET.get(case_id, {})
        fps      = budget.get("fps", self._gif_fps)
        all_frames = event.get("pre_frames", []) + event.get("post_frames", [])

        gif_path, thumb_path = self._evidence.compile_gif(
            frames=all_frames,
            case_id=case_id,
            camera_id=camera_id,
            fps=fps,
            quality=self._gif_quality,
        )

        # Append resolve record to JSONL
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
        """Draw coloured [CZ] / [KZ] mini-badges above each bounding box."""
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
        """Draw ROI zone overlays, zone badges, alert banners, case badge, counters, frame ID."""
        out = frame.copy()
        h, w = out.shape[:2]

        severity_color = {
            SEVERITY_NORMAL  : _CLR_NORMAL,
            SEVERITY_ALERT   : _CLR_ALERT,
            SEVERITY_CRITICAL: _CLR_CRITICAL,
        }.get(result.severity, _CLR_NORMAL)

        # Draw zone polygons with translucent fill
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

        # Draw [CZ] / [KZ] badges on each detection box
        self._draw_zone_badges(out, result.cashier_zone, result.customer_zone)

        # Alert banners (top-left)
        for i, alert in enumerate(result.alerts):
            cv2.putText(out, alert, (10, 28 + i * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, severity_color, 2, cv2.LINE_AA)

        # Case badge (bottom-right)
        badge = f"{result.case_id} | {result.severity}"
        (bw, bh), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
        cv2.rectangle(out, (w - bw - 14, h - bh - 16), (w - 2, h - 2), severity_color, -1)
        cv2.putText(out, badge, (w - bw - 10, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 2, cv2.LINE_AA)

        # Zone counters (bottom-left)
        for i, (label, zc) in enumerate((
            ("CZ", result.cashier_zone),
            ("KZ", result.customer_zone),
        )):
            line = f"{label}: P={zc.persons} D={zc.drawers} C={zc.cash}"
            cv2.putText(out, line, (8, h - 10 - (1 - i) * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        # Frame counter (bottom-right, above case badge)
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
        camera_id  = self._camera_id

        # 1. Push raw frame to per-camera buffer (before annotation)
        self._push_frame(camera_id, frame)
        self._frame_count += 1
        frame_id = self._frame_count

        # 2. Zone assignment + counts
        cz, kz = self._count_zones(detections, w, h)

        # 3. Business logic evaluation
        case_id, severity, alerts, transaction = self._evaluate(cz, kz, now)

        # 4. Detect event resolution (alert case changed)
        self._check_resolved(camera_id, case_id, now)

        # 5. Debounce gate
        debounced = self._check_debounce(camera_id, case_id)

        # 6. Build result
        result = CashierResult(
            cashier_zone  = cz,
            customer_zone = kz,
            case_id       = case_id,
            severity      = severity,
            alerts        = alerts,
            transaction   = transaction,
            frame_saved   = False,
            evidence_path = None,
            frame_id      = frame_id,
        )

        # 7. Annotate live frame
        if self.save:
            context["data"]["frame"] = self._draw(frame, result)

        annotated = context["data"]["frame"] if self.save else frame

        # 8. Evidence logic
        should_save = severity in (SEVERITY_ALERT, SEVERITY_CRITICAL)

        if should_save and debounced:
            if camera_id not in self._active_events:
                # Start new tracked alert event
                meta = self._build_meta(camera_id, case_id, severity, alerts, cz, kz, frame_id)
                img_path = self._evidence.save_frame(annotated, case_id, meta, camera_id, self._gif_quality)
                result.frame_saved   = True
                result.evidence_path = img_path
                meta["thumb_path"]   = img_path
                self._start_event(camera_id, case_id, meta, annotated, now)
            else:
                # Accumulate post-event frame
                self._accumulate_post(camera_id, case_id, annotated, now)

        elif case_id == "N3" and transaction:
            # Save a single transaction keyframe (no GIF)
            meta     = self._build_meta(camera_id, case_id, severity, alerts, cz, kz, frame_id)
            img_path = self._evidence.save_frame(annotated, case_id, meta, camera_id, self._gif_quality)
            self._evidence.append_log({**meta, "status": "triggered"})
            result.frame_saved   = True
            result.evidence_path = img_path

        elif camera_id in self._active_events:
            # Still inside a tracked event (could be transitioning)
            self._accumulate_post(camera_id, case_id, annotated, now)

        # 9. Publish to API state tracker
        try:
            from apis.cashier import push_result
            push_result(camera_id, result.to_dict())
        except Exception:
            pass  # API module not loaded in subprocess — silently skip

        # 10. Write to context
        context["data"]["use_case"]["cashier"] = result.to_dict()
        return context
