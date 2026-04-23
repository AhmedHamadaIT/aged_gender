"""
stream.py
---------
Frame sourcing — RTSP stream or local video.
Exposes a single generator: frames(source)

For RTSP: FFmpeg subprocess (TCP, discard corrupt, optional re-encode to H.264)
instead of raw cv2/HEVC decode.

frames(rtsp_url)   → RTSP via FFmpeg
frames(video_path) → local file via cv2
frames()           → uses USE_STREAM / RTSP from .env
"""

import json
import os
import shlex
import subprocess
import time
from typing import Iterator, List, Optional

import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from logger.logger_config import Logger

log = Logger.get_logger(__name__)

USE_STREAM = os.getenv("USE_STREAM", "True").lower() in ("true", "1", "yes")
RTSP_URL = os.getenv("CAMERA_1_URL", "")
INPUT_VIDEO = os.getenv("INPUT_VIDEO", "./videos/sample.mp4")

def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.lower() in ("true", "1", "yes", "on")


RTSP_PROFILE = os.getenv("RTSP_PROFILE", "balanced").lower()
FORCE_TRANSCODE = _env_bool("FORCE_TRANSCODE", True)
FRAME_MIN_VARIANCE = float(os.getenv("FRAME_MIN_VARIANCE", "8.0"))
MAX_CONSEC_FAILS = int(os.getenv("STREAM_MAX_FAILS", "15"))
RECONNECT_BASE_DELAY = float(os.getenv("STREAM_RECONNECT_DELAY", "2.0"))
RTSP_DISCARD_CORRUPT = _env_bool(
    "RTSP_DISCARD_CORRUPT",
    default=(RTSP_PROFILE in ("hardening", "balanced")),
)
RTSP_LOW_DELAY = _env_bool("RTSP_LOW_DELAY", default=True)
RTSP_FFMPEG_OPTIONS = os.getenv("RTSP_FFMPEG_OPTIONS", "").strip()
RTSP_ENABLE_GPU_DECODE = _env_bool("RTSP_ENABLE_GPU_DECODE", default=False)
RTSP_HWACCEL = os.getenv("RTSP_HWACCEL", "cuda").strip()
RTSP_HWDECODER = os.getenv("RTSP_HWDECODER", "").strip()
RTSP_HWACCEL_DEVICE = os.getenv("RTSP_HWACCEL_DEVICE", "").strip()
RTSP_TRANSPORT = os.getenv("RTSP_TRANSPORT", "tcp").strip() or "tcp"


def _parse_ladder() -> List[dict]:
    raw = os.getenv("QUALITY_LADDER", "1920x1080x10,1280x720x10,854x480x8")
    levels = []
    for entry in raw.split(","):
        parts = [p.strip() for p in entry.strip().split("x")]
        if len(parts) == 3:
            w, h, fps = int(parts[0]), int(parts[1]), int(parts[2])
            levels.append(
                {
                    "w": w,
                    "h": h,
                    "fps": fps,
                    "label": f"{h}p@{fps}fps",
                }
            )
    if not levels:
        levels = [{"w": 1920, "h": 1080, "fps": 10, "label": "1080p@10fps"}]
    return levels


QUALITY_LADDER = _parse_ladder()
DOWNGRADE_AFTER = int(os.getenv("STREAM_DOWNGRADE_AFTER", "20"))
DOWNGRADE_COOLDOWN = float(os.getenv("STREAM_DOWNGRADE_COOLDOWN", "30.0"))

# Updated by _rtsp_frames for monitoring (exposed to FrameBus)
_current_quality_label: str = QUALITY_LADDER[0]["label"] if QUALITY_LADDER else "live"
_current_decode_failures: int = 0
_current_decoder_type: str = "cpu"
_current_hw_decoder_requested: Optional[str] = None
_current_hw_decoder_active: bool = False
_current_profile: str = RTSP_PROFILE
_current_transport: str = RTSP_TRANSPORT

_redis_pub = None


def _get_redis_pub():
    global _redis_pub
    if _redis_pub is not None:
        return _redis_pub
    try:
        import redis

        url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        r = redis.Redis.from_url(url, socket_connect_timeout=2)
        r.ping()
        _redis_pub = r
    except Exception:
        _redis_pub = False
    return _redis_pub


def _publish_quality_event(camera_id: Optional[str], from_label: str, to_label: str) -> None:
    r = _get_redis_pub()
    if not r or r is False:
        return
    try:
        payload = {
            "event": "stream_quality_changed",
            "camera_id": camera_id or "unknown",
            "from": from_label,
            "to": to_label,
            "ts": time.time(),
        }
        r.publish("stream:quality_events", json.dumps(payload))
    except Exception:
        pass


def _is_valid_frame(frame: np.ndarray, min_variance: float = FRAME_MIN_VARIANCE) -> bool:
    if frame is None or frame.size == 0:
        return False
    if frame.ndim != 3 or frame.shape[2] != 3:
        return False
    small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return float(gray.var()) >= min_variance


class _FFmpegRTSPReader:
    """FFmpeg subprocess reading RTSP to raw BGR24."""

    def __init__(self, url: str, quality_level: int = 0):
        self.url = url
        self.quality_level = min(quality_level, len(QUALITY_LADDER) - 1) if QUALITY_LADDER else 0
        self._proc: Optional[subprocess.Popen] = None
        self._nbytes: int = 0
        self._gpu_fallback_disabled = False
        self.hw_decoder_requested: Optional[str] = RTSP_HWDECODER or None
        self.hw_decoder_active: bool = False
        self.decoder_type: str = "cpu"
        self._first_frame_ok: bool = False

    @property
    def _quality(self) -> dict:
        if not QUALITY_LADDER:
            return {"w": 1920, "h": 1080, "fps": 10, "label": "1080p@10fps"}
        return QUALITY_LADDER[self.quality_level]

    def _build_cmd(self, use_hwdecode: bool) -> list:
        q = self._quality
        w, h, fps = q["w"], q["h"], q["fps"]
        cmd = [
            "ffmpeg",
            "-loglevel",
            "error",
            "-rtsp_transport",
            RTSP_TRANSPORT,
        ]
        if RTSP_DISCARD_CORRUPT or FORCE_TRANSCODE:
            cmd.extend(["-fflags", "+discardcorrupt"])
        if RTSP_LOW_DELAY:
            cmd.extend(["-flags", "low_delay"])
        if use_hwdecode and RTSP_HWACCEL:
            cmd.extend(["-hwaccel", RTSP_HWACCEL])
            if RTSP_HWACCEL_DEVICE:
                cmd.extend(["-hwaccel_device", RTSP_HWACCEL_DEVICE])
            if RTSP_HWDECODER:
                cmd.extend(["-c:v", RTSP_HWDECODER])
        if RTSP_FFMPEG_OPTIONS:
            cmd.extend(shlex.split(RTSP_FFMPEG_OPTIONS))
        cmd.extend(
            [
                "-i",
                self.url,
                "-vf",
                f"scale={w}:{h},fps={fps}",
                "-pix_fmt",
                "bgr24",
                "-f",
                "rawvideo",
                "pipe:1",
            ]
        )
        return cmd

    def connect(self) -> None:
        q = self._quality
        self._nbytes = q["w"] * q["h"] * 3
        want_gpu = RTSP_ENABLE_GPU_DECODE and not self._gpu_fallback_disabled
        self.hw_decoder_requested = RTSP_HWDECODER or None
        self.hw_decoder_active = bool(want_gpu and RTSP_HWDECODER)
        self.decoder_type = "gpu" if self.hw_decoder_active else "cpu"
        cmd = self._build_cmd(use_hwdecode=want_gpu)
        log.info(f"[STREAM] FFmpeg cmd: {' '.join(cmd)}")
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=10**7,
        )
        time.sleep(0.15)
        if self._proc.poll() is not None and want_gpu:
            # Decoder not available (common mismatch across Jetson/x86 builds).
            self._gpu_fallback_disabled = True
            self.hw_decoder_active = False
            self.decoder_type = "cpu_fallback"
            log.warning("[STREAM] GPU decode unavailable, falling back to CPU decode.")
            cmd = self._build_cmd(use_hwdecode=False)
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=10**7,
            )
        log.info(f"[STREAM] FFmpeg PID {self._proc.pid} — {q['label']}")

    def read_frame(self) -> Optional[np.ndarray]:
        if self._proc is None or self._proc.stdout is None:
            return None
        raw = self._proc.stdout.read(self._nbytes)
        if len(raw) != self._nbytes:
            return None
        self._first_frame_ok = True
        if self.hw_decoder_requested and not self.hw_decoder_active:
            self.decoder_type = "cpu_fallback"
        q = self._quality
        return np.frombuffer(raw, dtype=np.uint8).reshape((q["h"], q["w"], 3))

    def release(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None


class _VideoReader:
    def __init__(self, path: str):
        self.path = path
        self.cap = None

    def connect(self) -> None:
        log.info(f"[VIDEO] Opening: {self.path}")
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise RuntimeError(f"[VIDEO] Cannot open: {self.path}")

    def read_frame(self) -> Optional[np.ndarray]:
        if self.cap is None:
            return None
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self) -> None:
        if self.cap:
            self.cap.release()
            self.cap = None


def frames(
    source: str = None,
    camera_id: Optional[str] = None,
) -> Iterator[np.ndarray]:
    """
    Generator yielding validated BGR numpy frames.

    Args:
        source: RTSP URL, video file path, or None (uses .env defaults)
        camera_id: optional — used for stream:quality_events Redis channel on downgrade
    """
    if source is None:
        source = RTSP_URL if USE_STREAM else INPUT_VIDEO

    is_rtsp = source.startswith("rtsp://")

    if not is_rtsp:
        yield from _video_frames(source)
        return

    yield from _rtsp_frames(source, camera_id=camera_id)


def _video_frames(path: str) -> Iterator[np.ndarray]:
    reader = _VideoReader(path)
    reader.connect()
    try:
        while True:
            frame = reader.read_frame()
            if frame is None:
                break
            if _is_valid_frame(frame):
                yield frame
    finally:
        reader.release()


def _rtsp_frames(url: str, camera_id: Optional[str] = None) -> Iterator[np.ndarray]:
    global _current_quality_label, _current_decode_failures
    global _current_decoder_type, _current_hw_decoder_requested, _current_hw_decoder_active
    quality_level = 0
    _current_quality_label = (
        QUALITY_LADDER[quality_level]["label"] if QUALITY_LADDER else "live"
    )
    _current_decode_failures = 0
    _current_decoder_type = "cpu"
    _current_hw_decoder_requested = RTSP_HWDECODER or None
    _current_hw_decoder_active = False
    last_downgrade = 0.0
    retry_delay = RECONNECT_BASE_DELAY
    consec_fails = 0
    consec_bad = 0

    while True:
        reader = _FFmpegRTSPReader(url, quality_level)
        try:
            reader.connect()
        except Exception as exc:
            log.error(f"[STREAM] FFmpeg spawn failed: {exc}")
            time.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 30.0)
            continue

        retry_delay = RECONNECT_BASE_DELAY
        if QUALITY_LADDER:
            _current_quality_label = QUALITY_LADDER[quality_level]["label"]
        _current_decoder_type = reader.decoder_type
        _current_hw_decoder_requested = reader.hw_decoder_requested
        _current_hw_decoder_active = reader.hw_decoder_active
        if _current_hw_decoder_requested and not _current_hw_decoder_active:
            _current_decoder_type = "cpu_fallback"
            log.warning(
                f"[STREAM] GPU decode requested ({_current_hw_decoder_requested}) "
                "but not active — falling back to CPU"
            )
        log.info(
            f"[STREAM] Connected — quality={_current_quality_label}, decoder={_current_decoder_type}"
        )

        while True:
            frame = reader.read_frame()
            if frame is None:
                consec_fails += 1
                _current_decode_failures += 1
                if consec_fails >= MAX_CONSEC_FAILS:
                    log.warning(
                        f"[STREAM] {consec_fails} consecutive pipe failures — reconnecting"
                    )
                    break
                continue

            consec_fails = 0

            if not _is_valid_frame(frame):
                consec_bad += 1
                _current_decode_failures += 1
                log.debug(
                    f"[STREAM] frame rejected (low variance) — bad streak: {consec_bad}"
                )
                now = time.time()
                if (
                    consec_bad >= DOWNGRADE_AFTER
                    and QUALITY_LADDER
                    and quality_level < len(QUALITY_LADDER) - 1
                    and now - last_downgrade > DOWNGRADE_COOLDOWN
                ):
                    from_l = QUALITY_LADDER[quality_level]["label"]
                    quality_level += 1
                    last_downgrade = now
                    consec_bad = 0
                    to_l = QUALITY_LADDER[quality_level]["label"]
                    _current_quality_label = to_l
                    log.warning(
                        f"[STREAM] downgrading to {to_l} after {DOWNGRADE_AFTER} bad frames"
                    )
                    _publish_quality_event(camera_id, from_l, to_l)
                    break
                continue

            consec_bad = 0
            _current_decoder_type = reader.decoder_type
            _current_hw_decoder_requested = reader.hw_decoder_requested
            _current_hw_decoder_active = reader.hw_decoder_active
            yield frame

        reader.release()
        log.info(f"[STREAM] Reconnecting in {retry_delay:.1f}s …")
        time.sleep(retry_delay)
        retry_delay = min(retry_delay * 1.5, 30.0)
