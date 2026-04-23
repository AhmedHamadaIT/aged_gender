"""
RTSP / OpenCV-FFmpeg capture options (TCP-only, shared by stream reader and camera snapshots).
"""
from __future__ import annotations

import os

import cv2


def build_rtsp_ffmpeg_options() -> str:
    """
    Build OPENCV_FFMPEG_CAPTURE_OPTIONS for RTSP over TCP.

    Format (OpenCV FFmpeg backend): key;value|key2;value2|...
    Always includes rtsp_transport;tcp (project default: TCP-only).

    Override or extend with RTSP_FFMPEG_EXTRA_OPTIONS (pipe-separated key;value segments),
    e.g. "max_delay;500000|fflags;nobuffer|reorder_queue_size;0".
    """
    segments: list[str] = ["rtsp_transport;tcp"]
    extra = os.getenv(
        "RTSP_FFMPEG_EXTRA_OPTIONS",
        "max_delay;500000|fflags;nobuffer|reorder_queue_size;0",
    )
    for part in extra.split("|"):
        part = part.strip()
        if not part:
            continue
        if ";" not in part:
            continue
        # Avoid duplicating transport if user re-specifies
        if part.lower().startswith("rtsp_transport;"):
            continue
        segments.append(part)
    return "|".join(segments)


def apply_rtsp_ffmpeg_env() -> str:
    """Set OPENCV_FFMPEG_CAPTURE_OPTIONS and return the options string used."""
    opts = build_rtsp_ffmpeg_options()
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = opts
    return opts


def _set_rtsp_capture_props(cap: cv2.VideoCapture) -> None:
    buf = max(1, int(os.getenv("RTSP_OPENCV_BUFFER_SIZE", "1")))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, buf)
    otm = int(os.getenv("RTSP_OPEN_TIMEOUT_MSEC", "15000"))
    rtm = int(os.getenv("RTSP_READ_TIMEOUT_MSEC", "0"))
    for prop, val in (
        (getattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC", -1), otm),
        (getattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC", -1), rtm),
    ):
        if prop != -1 and val > 0:
            cap.set(prop, val)


def open_rtsp_videocapture(url: str) -> cv2.VideoCapture:
    """
    Open an RTSP URL with TCP-only FFmpeg options and low-latency-style defaults.
    """
    apply_rtsp_ffmpeg_env()
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    _set_rtsp_capture_props(cap)
    return cap
