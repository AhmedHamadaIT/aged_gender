"""
utils.py
--------
Drawing, resizing, and frame saving utilities.
"""

import os
import cv2
import numpy as np
from typing import List
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
def save_frame(frame: np.ndarray, output_dir: str, frame_count: int, prefix: str = "frame") -> str:
    """
    Save a single frame as JPEG to output_dir.

    Args:
        frame:       BGR numpy array
        output_dir:  directory to save into (created if not exists)
        frame_count: used to generate filename
        prefix:      filename prefix (default: 'frame')

    Returns:
        path of saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{prefix}_{frame_count:06d}.jpg"
    path     = os.path.join(output_dir, filename)
    cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return path