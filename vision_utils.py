from __future__ import annotations

"""
vision_utils.py
----------------
Drawing, resizing, and frame saving utilities.
ML Image Contract V2 helpers (get_base_url, build_image, make_evidence_paths).
"""

import logging
import os
import uuid as _uuid
import cv2
import numpy as np
from datetime import datetime, timezone
from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from services.detector import Detection


# ─────────────────────────────────────────────
# Color palette (BGR)
# ─────────────────────────────────────────────
COLORS = [
    ( 56, 193, 114),
    ( 52, 152, 219),
    (231,  76,  60),
    (241, 196,  15),
    (155,  89, 182),
    ( 26, 188, 156),
    (230, 126,  34),
    ( 52,  73,  94),
]


# ─────────────────────────────────────────────
# Drawing
# ─────────────────────────────────────────────
def draw_detections(frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
    """Draw bounding boxes and labels onto a copy of the frame."""
    out = frame.copy()
    for det in detections:
        color = COLORS[det.class_id % len(COLORS)]
        label = f"{det.class_name} {det.confidence:.2f}"

        cv2.rectangle(out, (det.x1, det.y1), (det.x2, det.y2), color, 2)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(out, (det.x1, det.y1 - th - 8), (det.x1 + tw + 4, det.y1), color, -1)
        cv2.putText(
            out, label, (det.x1 + 2, det.y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (255, 255, 255), 1, cv2.LINE_AA,
        )
    return out


def draw_overlay(frame: np.ndarray, fps: float, mode: str, count: int) -> np.ndarray:
    """Draw FPS / mode / count overlay in the top-left corner."""
    lines = [
        f"Mode   : {mode}",
        f"FPS    : {fps:.1f}",
        f"Objects: {count}",
    ]
    y = 28
    for text in lines:
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0),       3, cv2.LINE_AA)
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        y += 26
    return frame


def annotate(frame: np.ndarray, detections: List[Detection], fps: float, mode: str) -> np.ndarray:
    """Draw detections + overlay in one call. Returns annotated copy."""
    frame = draw_detections(frame, detections)
    frame = draw_overlay(frame, fps, mode, len(detections))
    return frame


# ─────────────────────────────────────────────
# Resizing
# ─────────────────────────────────────────────
def resize(frame: np.ndarray, width: int, height: int = 0) -> np.ndarray:
    """
    Resize frame to given width (and optional height).
    If height=0, preserves aspect ratio from width alone.
    No-op if frame is already at or below the target size.
    """
    h, w = frame.shape[:2]
    if height == 0:
        if w <= width:
            return frame
        height = int(h * width / w)
    return cv2.resize(frame, (width, height))


def resize_for_display(frame: np.ndarray, max_width: int = 1280) -> np.ndarray:
    """Resize frame to fit within max_width, preserving aspect ratio."""
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    return cv2.resize(frame, (max_width, int(h * max_width / w)))


# ─────────────────────────────────────────────
# Saving
# ─────────────────────────────────────────────
def _env_int_clamped(name: str, default: int, lo: int, hi: int) -> int:
    try:
        v = int(os.getenv(name, str(default)))
    except ValueError:
        v = default
    return max(lo, min(hi, v))


def save_frame(
    frame: np.ndarray,
    output_dir: str,
    frame_count: int,
    prefix: str = "frame",
    *,
    jpeg_quality: int | None = None,
    save_format: str | None = None,
) -> str:
    """
    Save a single frame to output_dir.

    Format and quality are controlled by SAVE_FORMAT (``jpg`` | ``jpeg`` | ``webp``)
    and SAVE_JPEG_QUALITY / SAVE_WEBP_QUALITY unless overridden per-call.

    Args:
        frame:       BGR numpy array
        output_dir:  directory to save into (created if not exists)
        frame_count: used to generate filename
        prefix:      filename prefix (default: 'frame')
        jpeg_quality: optional JPEG quality 1–100 (default from SAVE_JPEG_QUALITY, default 80)
        save_format: optional ``jpg`` or ``webp`` (default from SAVE_FORMAT env)

    Returns:
        path of saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    fmt = (save_format or os.getenv("SAVE_FORMAT", "jpg")).lower().strip()
    if fmt in ("jpeg",):
        fmt = "jpg"
    ext = ".webp" if fmt == "webp" else ".jpg"
    filename = f"{prefix}_{frame_count:06d}{ext}"
    path = os.path.join(output_dir, filename)

    if fmt == "webp":
        q = jpeg_quality if jpeg_quality is not None else _env_int_clamped(
            "SAVE_WEBP_QUALITY", 80, 1, 100
        )
        cv2.imwrite(path, frame, [cv2.IMWRITE_WEBP_QUALITY, q])
    else:
        q = jpeg_quality if jpeg_quality is not None else _env_int_clamped(
            "SAVE_JPEG_QUALITY", 80, 1, 100
        )
        cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, q])
    return path


def dir_disk_usage_bytes(root: str) -> int:
    """Total size of files under root (recursive). Missing dir → 0."""
    total = 0
    if not root or not os.path.isdir(root):
        return 0
    try:
        for dirpath, _dirnames, filenames in os.walk(root):
            for fn in filenames:
                fp = os.path.join(dirpath, fn)
                try:
                    total += os.path.getsize(fp)
                except OSError:
                    pass
    except OSError:
        pass
    return total


def log_evidence_dirs_disk_usage(logger: logging.Logger, label: str = "storage") -> None:
    """Log summed byte sizes for common evidence / output roots (best-effort)."""
    roots = {
        "OUTPUT_DIR": os.getenv("OUTPUT_DIR", "./outputs"),
        "CAPTURE_DIR": os.getenv("CAPTURE_DIR", "./evidence/capture"),
        "SCENE_DIR": os.getenv("SCENE_DIR", "./evidence/scene"),
        "GALLERY_DIR": os.getenv("GALLERY_DIR", "/local/storage/gallery"),
        "CASHIER_EVIDENCE_DIR": os.getenv("CASHIER_EVIDENCE_DIR", "./evidence/cashier"),
    }
    parts = []
    for name, p in roots.items():
        b = dir_disk_usage_bytes(p)
        parts.append(f"{name}={p} ({b} bytes)")
    logger.info("[%s] disk usage by directory: %s", label, "; ".join(parts))


# ─────────────────────────────────────────────
# ML Image Contract V2
# ─────────────────────────────────────────────
def get_base_url() -> str:
    """Resolve public base URL for image links.

    Priority:
    1) PUBLIC_ML_BASE_URL (legacy contract)
    2) CAMERA_SNAPSHOT_BASE_URL (shared API base)
    3) http://127.0.0.1:9000 (safe local default)
    """
    return (
        os.getenv("PUBLIC_ML_BASE_URL")
        or os.getenv("CAMERA_SNAPSHOT_BASE_URL")
        or "http://127.0.0.1:9000"
    ).rstrip("/")


def _sanitize_rel_path(path: str) -> str:
    """
    Normalize an arbitrary path to a clean relative path suitable for URL embedding.
    - Strips leading slashes
    - Removes known absolute prefixes (app/evidence/, evidence/)
    - Collapses double-slashes
    """
    path = path.lstrip("/")
    for prefix in ("app/evidence/", "evidence/"):
        if path.startswith(prefix):
            path = path[len(prefix) :]
            break
    path = path.replace("//", "/")
    return path


def build_image(path: str, img_type: str) -> dict:
    """Build a standard V2 image object. path is sanitized internally."""
    rel = _sanitize_rel_path(path)
    base = get_base_url()
    url = f"{base}/evidence/{rel}"
    return {
        "url": url,
        "path": rel,
        "type": img_type,
        "format": "image/jpeg",
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def draw_evidence_scene(
    frame: np.ndarray,
    *,
    subject_bbox: Tuple[int, int, int, int],
    label: str = "",
    secondary_bbox: Tuple[int, int, int, int] = None,
    secondary_label: str = "",
    zone_points: list = None,
    line_endpoints: Tuple[Tuple[int, int], Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Return an annotated copy of *frame* for use as scene evidence.

    Draws (in order, so labels are on top):
    * ``line_endpoints`` — cyan counting/virtual line
    * ``zone_points`` — yellow filled-edge polygon zone
    * ``subject_bbox`` — green rect + ``label`` (person / face)
    * ``secondary_bbox`` — orange rect + ``secondary_label`` (phone, alarm item…)

    All parameters except *frame* and *subject_bbox* are optional.
    """
    vis = frame.copy()

    # Virtual line (e.g. cross-line)
    if line_endpoints is not None:
        cv2.line(vis, line_endpoints[0], line_endpoints[1], (0, 255, 255), 2, cv2.LINE_AA)

    # Zone polygon (e.g. PPE / phone zone)
    if zone_points and len(zone_points) >= 3:
        pts = np.array(
            [(int(p["x"]), int(p["y"])) for p in zone_points], dtype=np.int32
        ).reshape((-1, 1, 2))
        cv2.polylines(vis, [pts], isClosed=True, color=(0, 220, 255), thickness=2,
                      lineType=cv2.LINE_AA)

    # Primary subject box + label
    x1, y1, x2, y2 = (int(v) for v in subject_bbox)
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 220, 0), 2, cv2.LINE_AA)
    if label:
        cv2.putText(vis, label, (x1, max(18, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

    # Secondary object box + label (e.g. phone)
    if secondary_bbox is not None:
        sx1, sy1, sx2, sy2 = (int(v) for v in secondary_bbox)
        cv2.rectangle(vis, (sx1, sy1), (sx2, sy2), (0, 140, 255), 2, cv2.LINE_AA)
        if secondary_label:
            cv2.putText(vis, secondary_label, (sx1, max(18, sy1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 180, 255), 1, cv2.LINE_AA)

    return vis


def make_evidence_paths(camera_id: str, event_id: str) -> Tuple[str, str]:
    """
    Return (capture_relative_path, scene_relative_path) using V2 naming convention:
    {YYYY-MM-DD}/{camera_id}_{event_id}_{uuid8}.jpg
    """
    date = datetime.now().strftime("%Y-%m-%d")
    cap_uuid = _uuid.uuid4().hex[:8]
    scene_uuid = _uuid.uuid4().hex[:8]
    safe_cam = camera_id.strip() or "unknown_camera"
    return (
        f"{date}/{safe_cam}_{event_id}_{cap_uuid}.jpg",
        f"{date}/{safe_cam}_{event_id}_{scene_uuid}.jpg",
    )