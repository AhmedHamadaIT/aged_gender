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
from utils.rtsp_ffmpeg import apply_rtsp_ffmpeg_env, open_rtsp_videocapture, warmup_rtsp_capture

log = Logger.get_logger(__name__)

USE_STREAM  = os.getenv("USE_STREAM",   "True").lower() in ("true", "1", "yes")
RTSP_URL    = os.getenv("CAMERA_1_URL", "")
INPUT_VIDEO = os.getenv("INPUT_VIDEO",  "./videos/sample.mp4")

# Reconnect + decode health
RTSP_MAX_CONSECUTIVE_FAILS = max(1, int(os.getenv("RTSP_MAX_CONSECUTIVE_READ_FAILS", "10")))
STREAM_RECONNECT_BASE_SEC  = max(0.1, float(os.getenv("STREAM_RECONNECT_BASE_SEC", "2")))
STREAM_RECONNECT_MAX_SEC   = max(STREAM_RECONNECT_BASE_SEC, float(os.getenv("STREAM_RECONNECT_MAX_SEC", "30")))


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except ValueError:
        return max(minimum, default)


def _env_float(name: str, default: float, minimum: float = 0.0) -> float:
    try:
        return max(minimum, float(os.getenv(name, str(default))))
    except ValueError:
        return max(minimum, default)


class _FrameThrottle:
    def __init__(
        self,
        frame_skip: int = 1,
        target_fps: float = 0.0,
        time_fn=time.monotonic,
    ):
        self.frame_skip = max(1, frame_skip)
        self.target_fps = max(0.0, target_fps)
        self._min_interval = 1.0 / self.target_fps if self.target_fps > 0 else 0.0
        self._time_fn = time_fn
        self._seen = 0
        self._last_yield_at = None

    @classmethod
    def from_env(cls):
        return cls(
            frame_skip=_env_int("FRAME_SKIP", 1, minimum=1),
            target_fps=_env_float("STREAM_TARGET_FPS", 0.0, minimum=0.0),
        )

    def should_yield(self) -> bool:
        self._seen += 1
        if self.frame_skip > 1 and self._seen % self.frame_skip != 0:
            return False

        if self._min_interval <= 0:
            return True

        now = self._time_fn()
        if self._last_yield_at is not None and now - self._last_yield_at < self._min_interval:
            return False

        self._last_yield_at = now
        return True


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
# Cumulative failed VideoCapture.read() attempts (RTSP gaps, decode hiccups, EOS on file).
_current_decode_failures = 0
# Successful RTSP reconnects after a failure burst (initial connect is not counted).
_rtsp_reconnect_count = 0
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
        wup = max(0, int(os.getenv("RTSP_WARMUP_FRAMES", "8")))
        okw = warmup_rtsp_capture(self.cap)
        w   = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 0.0
        log.info("[STREAM] Connected — %dx%d @ %.1ffps", w, h, fps)
        w_tgt = int(os.getenv("WIDTH", "1280"))
        h_raw = int(os.getenv("HEIGHT", "0"))
        if h_raw > 0:
            log.info(
                "[STREAM] Processing size target: WIDTH=%d HEIGHT=%d (set in env; FrameBus resizes to this).",
                w_tgt,
                h_raw,
            )
        else:
            log.info(
                "[STREAM] Processing width target: WIDTH=%d (HEIGHT=0 keep aspect; FrameBus resizes). "
                "If CPU load is high, use a 720p/1080p substream or lower WIDTH.",
                w_tgt,
            )
        if wup > 0 and okw < wup:
            log.warning(
                "[STREAM] Warmup only got %d/%d frames; decode may be unstable (HEVC/RTSP).",
                okw,
                wup,
            )

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
    global _current_decode_failures, _rtsp_reconnect_count

    if source is None:
        source = RTSP_URL if USE_STREAM else INPUT_VIDEO

    is_rtsp = source.startswith("rtsp://")
    reader  = _RTSPReader(source) if is_rtsp else _VideoReader(source)

    consecutive_fails = 0
    reconnect_streak  = 0
    throttle = _FrameThrottle.from_env()

    reader.connect()
    if throttle.frame_skip > 1 or throttle.target_fps > 0:
        log.info(
            "[STREAM] Frame throttle active — FRAME_SKIP=%d STREAM_TARGET_FPS=%.2f",
            throttle.frame_skip,
            throttle.target_fps,
        )
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
                            _rtsp_reconnect_count += 1
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
            if not throttle.should_yield():
                continue
            yield frame
    finally:
        reader.release()
