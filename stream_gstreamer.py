"""
Jetson / L4T RTSP ingest via GStreamer + OpenCV VideoCapture (CAP_GSTREAMER).

Uses hardware NVDEC (nvv4l2decoder) and nvvidconv to BGR in system memory for
the rest of the Python/YOLO pipeline. Requires an OpenCV build with GStreamer
support (L4T base or Jetson Docker images; do not override with PyPI
opencv-python* without GStreamer).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import cv2

from stream_adapter import CameraProfile, is_jetson, profile_health

logger = logging.getLogger("shared_logger.stream_gstreamer")


def escape_gstreamer_url(url: str) -> str:
    """Escape double quotes in RTSP URL for GStreamer property strings."""
    return url.replace("\\", "\\\\").replace('"', '\\"')


def gstreamer_backend_available() -> bool:
    """True if this OpenCV build exposes CAP_GSTREAMER."""
    if not hasattr(cv2, "CAP_GSTREAMER"):
        return False
    try:
        build = cv2.getBuildInformation()
    except Exception:
        return True
    for line in build.splitlines():
        if "GStreamer:" in line:
            return "YES" in line.upper() or "yes" in line
    return False


def should_attempt_gstreamer(rtsp_backend: str) -> bool:
    """
    Whether connect() may try the GStreamer path.

    - gstreamer: always (caller validates Jetson + codec).
    - auto: on Jetson when RTSP_GSTREAMER is true and OpenCV has GStreamer.
    - ffmpeg / opencv: do not use GStreamer.
    """
    mode = (rtsp_backend or "auto").lower().strip()
    if mode in ("ffmpeg", "opencv"):
        return False
    if mode == "gstreamer":
        return True
    if not _env_flag("RTSP_GSTREAMER", "true"):
        return False
    if not is_jetson():
        return False
    if not gstreamer_backend_available():
        logger.info("[GST] OpenCV GStreamer backend not available; skipping")
        return False
    return True


def _env_flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).lower() in ("true", "1", "yes", "on")


def _rtsp_latency_ms() -> int:
    try:
        return max(0, int(os.getenv("RTSP_GST_LATENCY_MS", "0")))
    except ValueError:
        return 0


def _codec_uses_h265(codec: str) -> bool:
    c = (codec or "").lower()
    return c in ("hevc", "h265", "hev1", "hvc1")


def _codec_uses_h264(codec: str) -> bool:
    c = (codec or "").lower()
    return c in ("h264", "avc", "avc1", "h264v")


def build_gstreamer_pipeline(
    url: str,
    codec: str,
    width: int,
    height: int,
    transport: str | None = None,
) -> str:
    """
    Build a low-latency RTSP BGR appsink pipeline for Jetson (nvv4l2decoder).

    Raises:
        ValueError: if the probed codec is not H.264 / H.265.
    """
    t = (transport or os.getenv("RTSP_TRANSPORT", "tcp") or "tcp").lower()
    if _codec_uses_h265(codec):
        depay = "rtph265depay ! h265parse"
    elif _codec_uses_h264(codec):
        depay = "rtph264depay ! h264parse"
    else:
        raise ValueError(f"GStreamer path unsupported codec: {codec!r}")

    safe = escape_gstreamer_url(url)
    lat = _rtsp_latency_ms()
    # protocols=0x4 — TCP; reduces jitter on most LAN NVRs.
    rtspsrc = f'rtspsrc location="{safe}" latency={lat}'
    if t == "tcp":
        rtspsrc += " protocols=4"
    rtspsrc += " ! "

    extra = os.getenv("RTSP_GST_EXTRA", "").strip()
    if extra and not extra.endswith("!"):
        extra = extra + " ! "

    # Optional: enable-max-performance=1 on Jetson for NVDEC
    dec = os.getenv("RTSP_GST_NVV4L2DEC_EXTRAS", "enable-max-performance=1")
    if dec:
        nvv4l2 = f"nvv4l2decoder {dec} ! "
    else:
        nvv4l2 = "nvv4l2decoder ! "

    w = max(1, int(width))
    h = max(1, int(height))

    tail = (
        f"{depay} ! {nvv4l2}"
        f"nvvidconv ! video/x-raw, width={w}, height={h}, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! "
        f"appsink drop=true sync=false max-buffers=1"
    )
    if extra:
        return rtspsrc + extra + tail
    return rtspsrc + tail


def open_gstreamer_capture(profile: CameraProfile) -> Optional[tuple[cv2.VideoCapture, dict[str, Any]]]:
    """
    Open cv2.CAP_GSTREAMER with a pipeline from ``profile`` (codec + target size).

    Returns:
        (VideoCapture, health_dict) or None if open failed.
    """
    try:
        pipeline = build_gstreamer_pipeline(
            profile.url,
            profile.codec,
            profile.target_width,
            profile.target_height,
        )
    except ValueError as exc:
        logger.warning("[%s] %s", profile.camera_id, exc)
        return None

    try:
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[%s] GStreamer VideoCapture error: %s", profile.camera_id, exc)
        return None

    if not cap.isOpened():
        try:
            cap.release()
        except Exception:
            pass
        return None

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or profile.target_width
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or profile.target_height
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    hmap = profile_health(profile)
    hmap["codec"] = profile.codec
    hmap["native"] = f"{profile.native_width}x{profile.native_height}@{profile.native_fps}fps"
    hmap["target"] = f"{w}x{h}@gstreamer"
    hmap["decoder"] = "nvv4l2decoder"
    hmap["hw_decoder_requested"] = "nvv4l2decoder"
    hmap["hw_decoder_active"] = True
    hmap["backend"] = "gstreamer"
    hmap["healthy"] = True
    hmap["gstreamer_pipeline_hint"] = "nvv4l2decoder+appsink"
    hmap["native_open"] = f"{w}x{h}@{round(fps, 2)}fps"
    return cap, hmap
