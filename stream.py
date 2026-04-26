"""
stream.py
---------
Frame sourcing — RTSP stream or local video.
Exposes a single generator: frames(source)

frames(rtsp_url)   → reads directly from RTSP
frames(video_path) → reads from local video file
frames()           → uses USE_STREAM / RTSP_URL_1 from .env
"""

import os
import time

import cv2
from dotenv import load_dotenv

load_dotenv()

from logger.logger_config import Logger
from utils.rtsp_ffmpeg import apply_rtsp_ffmpeg_env, open_rtsp_videocapture

log = Logger.get_logger(__name__)

USE_STREAM  = os.getenv("USE_STREAM",   "True").lower() in ("true", "1", "yes")
RTSP_URL    = os.getenv("CAMERA_1_URL", "")
INPUT_VIDEO = os.getenv("INPUT_VIDEO",  "./videos/sample.mp4")

# Reconnect + decode health
RTSP_MAX_CONSECUTIVE_FAILS = max(1, int(os.getenv("RTSP_MAX_CONSECUTIVE_READ_FAILS", "10")))
STREAM_RECONNECT_BASE_SEC  = max(0.1, float(os.getenv("STREAM_RECONNECT_BASE_SEC", "2")))
STREAM_RECONNECT_MAX_SEC   = max(STREAM_RECONNECT_BASE_SEC, float(os.getenv("STREAM_RECONNECT_MAX_SEC", "30")))


def _parse_quality_ladder(raw: str) -> list[dict[str, int | str]]:
    ladder: list[dict[str, int | str]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.lower().split("x")
        if len(parts) != 3:
            ladder.append({"label": item})
            continue
        try:
            width, height, fps = (int(part) for part in parts)
        except ValueError:
            ladder.append({"label": item})
            continue
        ladder.append(
            {
                "width": width,
                "height": height,
                "fps": fps,
                "label": f"{width}x{height}@{fps}fps",
            }
        )
    return ladder


QUALITY_LADDER = _parse_quality_ladder(
    os.getenv("QUALITY_LADDER", "1920x1080x10,1280x720x10,854x480x8")
)
_current_quality_label = QUALITY_LADDER[0]["label"] if QUALITY_LADDER else "live"
_current_decode_failures = 0
_current_decoder_type = "cpu"
_current_hw_decoder_requested = (
    os.getenv("RTSP_HWDECODER")
    if os.getenv("RTSP_ENABLE_GPU_DECODE", "false").lower() in ("true", "1", "yes")
    else None
)
_current_hw_decoder_active = False
_current_profile = os.getenv("RTSP_PROFILE", "balanced").lower()
_current_transport = os.getenv("RTSP_TRANSPORT", "tcp")


class _RTSPReader:
    def __init__(self, url: str):
        self.url = url
        self.cap = None
        self._connect_count = 0

    def connect(self) -> None:
        self._connect_count += 1
        opts = apply_rtsp_ffmpeg_env()
        log.info(
            "[STREAM] Connecting (attempt %d): %s | ffmpeg_opts=%r",
            self._connect_count,
            self.url,
            opts,
        )
        self.release()
        self.cap = open_rtsp_videocapture(self.url)
        if not self.cap.isOpened():
            raise RuntimeError(f"[STREAM] Cannot open: {self.url}")
        w   = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 0.0
        log.info("[STREAM] Connected — %dx%d @ %.1ffps", w, h, fps)

    def read_frame(self):
        if self.cap is None:
            return None
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class _VideoReader:
    def __init__(self, path: str):
        self.path = path
        self.cap  = None

    def connect(self) -> None:
        log.info(f"[VIDEO] Opening: {self.path}")
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise RuntimeError(f"[VIDEO] Cannot open: {self.path}")

    def read_frame(self):
        if self.cap is None:
            return None
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def _reconnect_delay_seconds(attempt_index: int) -> float:
    # attempt_index: 0 after first failure batch, then increases
    exp = min(attempt_index, 8)
    delay = STREAM_RECONNECT_BASE_SEC * (2**exp)
    return min(STREAM_RECONNECT_MAX_SEC, delay)


def frames(source: str = None, camera_id: str = None):
    """
    Generator yielding BGR numpy frames.

    Args:
        source: RTSP URL, video file path, or None (uses .env defaults)
        camera_id: optional caller context for compatibility with FrameBus.
    """
    global _current_decode_failures

    if source is None:
        source = RTSP_URL if USE_STREAM else INPUT_VIDEO

    is_rtsp = source.startswith("rtsp://")
    reader  = _RTSPReader(source) if is_rtsp else _VideoReader(source)

    consecutive_fails = 0
    reconnect_streak  = 0

    reader.connect()
    try:
        while True:
            frame = reader.read_frame()
            if frame is None:
                consecutive_fails += 1
                _current_decode_failures += 1
                if consecutive_fails >= RTSP_MAX_CONSECUTIVE_FAILS:
                    if is_rtsp:
                        wait = _reconnect_delay_seconds(reconnect_streak)
                        log.info(
                            "[STREAM] %d consecutive read failures — reconnecting in %.1fs (streak=%d)",
                            consecutive_fails,
                            wait,
                            reconnect_streak,
                        )
                        reader.release()
                        time.sleep(wait)
                        try:
                            reader.connect()
                        except Exception as exc:  # noqa: BLE001
                            log.warning(
                                "[STREAM] Reconnect failed: %s — will retry with backoff", exc
                            )
                        reconnect_streak += 1
                        consecutive_fails = 0
                    else:
                        break  # end of video file
                continue
            consecutive_fails = 0
            reconnect_streak  = 0
            yield frame
    finally:
        reader.release()
