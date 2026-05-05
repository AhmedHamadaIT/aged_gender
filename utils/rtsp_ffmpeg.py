"""
RTSP / OpenCV-FFmpeg capture options (TCP-only, shared by stream reader and camera snapshots).
"""
from __future__ import annotations

import os

import cv2

# Universal “tolerant” preset when RTSP_FFMPEG_EXTRA_OPTIONS is unset (balanced / unknown codec).
# TCP is always forced separately via rtsp_transport;tcp.
_STABLE_EXTRA_DEFAULT = (
    "fflags;+genpts+discardcorrupt|max_delay;3000000|reorder_queue_size;512"
)
_LOW_LATENCY_EXTRA_DEFAULT = (
    "max_delay;500000|fflags;nobuffer|reorder_queue_size;0"
)


def _truthy(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).lower() in ("true", "1", "yes")


def _default_ffmpeg_extra_segments() -> list[str]:
    """
    Pick FFmpeg option segments when RTSP_FFMPEG_EXTRA_OPTIONS is unset.

    Priority:
    - RTSP_FFMPEG_OPTIONS (compose historically used this name): if set, treat as pipe-separated extras.
    - RTSP_LOW_DELAY=true → aggressive low-latency preset (may break some HEVC / B-frame RTSP sources).
    - RTSP_PROFILE=performance → same as low-latency.
    - Otherwise → stable preset (better tolerance across codecs / jittery RTSP).
    """
    legacy = os.getenv("RTSP_FFMPEG_OPTIONS")
    if legacy is not None and legacy.strip():
        return [p.strip() for p in legacy.split("|") if p.strip() and ";" in p.strip()]

    profile = os.getenv("RTSP_PROFILE", "balanced").lower()
    if _truthy("RTSP_LOW_DELAY", "false") or profile == "performance":
        return [p for p in _LOW_LATENCY_EXTRA_DEFAULT.split("|") if p]

    return [p for p in _STABLE_EXTRA_DEFAULT.split("|") if p]


def build_rtsp_ffmpeg_options() -> str:
    """
    Build OPENCV_FFMPEG_CAPTURE_OPTIONS for RTSP over TCP.

    Format (OpenCV FFmpeg backend): key;value|key2;value2|...
    Always includes rtsp_transport;tcp (project default: TCP-only).
    Always includes loglevel;error to suppress decoder warning noise.

    Override or extend with RTSP_FFMPEG_EXTRA_OPTIONS (pipe-separated key;value segments),
    e.g. "max_delay;500000|fflags;nobuffer|reorder_queue_size;0".

    If RTSP_FFMPEG_EXTRA_OPTIONS is unset, a preset is chosen from RTSP_PROFILE / RTSP_LOW_DELAY /
    RTSP_FFMPEG_OPTIONS (see _default_ffmpeg_extra_segments).
    """
    segments: list[str] = ["rtsp_transport;tcp", "loglevel;error"]
    extra_env = os.getenv("RTSP_FFMPEG_EXTRA_OPTIONS")
    if extra_env is None:
        parts = _default_ffmpeg_extra_segments()
    else:
        parts = [p.strip() for p in extra_env.split("|") if p.strip()]

    # Default on: bad/corrupt NALs are dropped instead of wedging the decoder (override with false).
    discard = _truthy("RTSP_DISCARD_CORRUPT", "true")

    fflag_tokens: list[str] = []
    other: list[str] = []
    has_discard_token = False
    seen_fflags: set[str] = set()

    def _merge_fflags_value(raw: str) -> None:
        nonlocal has_discard_token
        for tok in raw.replace(",", "+").split("+"):
            t = tok.strip()
            if not t:
                continue
            key = t.lower().lstrip("+")
            if key == "discardcorrupt":
                has_discard_token = True
            if key in seen_fflags:
                continue
            seen_fflags.add(key)
            fflag_tokens.append(t)

    for part in parts:
        if ";" not in part:
            continue
        low_part = part.lower()
        if low_part.startswith("rtsp_transport;"):
            continue
        if low_part.startswith("fflags;"):
            _merge_fflags_value(part.split(";", 1)[1])
        else:
            other.append(part)

    if discard and not has_discard_token:
        if "discardcorrupt" not in [x.lstrip("+").lower() for x in fflag_tokens]:
            fflag_tokens.append("discardcorrupt")

    if not discard:
        fflag_tokens = [
            t
            for t in fflag_tokens
            if t.lstrip("+").lower() != "discardcorrupt"
        ]

    if fflag_tokens:
        segments.append("fflags;" + "+".join(fflag_tokens))

    segments.extend(other)
    return "|".join(segments)


def apply_rtsp_ffmpeg_env() -> str:
    """Set OPENCV_FFMPEG_CAPTURE_OPTIONS and return the options string used."""
    opts = build_rtsp_ffmpeg_options()
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = opts
    return opts


def _set_rtsp_capture_props(cap: cv2.VideoCapture) -> None:
    # Slightly larger default buffer helps HEVC / B-frame RTSP; override per deployment.
    buf = max(1, int(os.getenv("RTSP_OPENCV_BUFFER_SIZE", "3")))
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
    Open an RTSP URL with TCP-only FFmpeg options (see build_rtsp_ffmpeg_options).
    """
    apply_rtsp_ffmpeg_env()
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    _set_rtsp_capture_props(cap)
    return cap


def warmup_rtsp_capture(cap: cv2.VideoCapture) -> int:
    """
    Read and discard a few frames after connect so decoders can stabilize (SPS/PPS/GOP).

    Controlled by RTSP_WARMUP_FRAMES (default 8). Returns number of frames successfully read.
    """
    n = max(0, int(os.getenv("RTSP_WARMUP_FRAMES", "8")))
    ok = 0
    for _ in range(n):
        ret, _frame = cap.read()
        if not ret:
            break
        ok += 1
    return ok
