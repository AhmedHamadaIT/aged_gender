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
from stream_adapter import AdaptiveStream, StreamProber, profile_health
from stream_gstreamer import gstreamer_backend_available, open_gstreamer_capture, should_attempt_gstreamer
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


def _env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).lower() in ("true", "1", "yes", "on")


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
_current_stream_health = {}
_current_rtsp_backend = "unknown"


def _rtsp_backend_mode() -> str:
    v = os.getenv("RTSP_BACKEND", "auto").lower().strip()
    if v in ("auto", "gstreamer", "ffmpeg", "opencv"):
        return v
    return "auto"


def _codec_ok_for_gstreamer(codec: str) -> bool:
    c = (codec or "").lower()
    return c in (
        "h264", "avc", "avc1", "h264v", "hevc", "h265", "hev1", "hvc1",
    )


class _RTSPReader:
    def __init__(self, url: str, camera_id: str | None = None):
        self.url = url
        self.camera_id = camera_id or "camera"
        self.cap = None
        self._adaptive = None
        self._adaptive_profile = None
        self._connect_count = 0
        self._force_cpu_decode = False
        self._disable_adaptive_ffmpeg = False
        self._adaptive_failure_cycles = 0
        self._ingest_backend: str | None = None
        self._gstreamer_profile = None  # probed profile used for GStreamer + metrics

    def connect(self) -> None:
        global _current_quality_label, _current_decoder_type, _current_hw_decoder_requested
        global _current_hw_decoder_active, _current_profile, _current_transport, _current_stream_health
        global _current_rtsp_backend

        self._connect_count += 1
        self.release()
        _current_profile = os.getenv("RTSP_PROFILE", "balanced").lower()
        _current_transport = os.getenv("RTSP_TRANSPORT", "tcp")
        rtsp_mode = _rtsp_backend_mode()
        # Skip PyPI opencv+GStreamer check on forced mode before probe
        if rtsp_mode == "gstreamer" and not gstreamer_backend_available():
            raise RuntimeError(
                "[STREAM] RTSP_BACKEND=gstreamer but OpenCV has no GStreamer support "
                "(use Jetson/L4T OpenCV build; do not override with opencv-python*)."
            )

        adaptive_wanted = (
            _env_bool("RTSP_ADAPTIVE_FFMPEG", "true")
            and not self._disable_adaptive_ffmpeg
            and rtsp_mode != "opencv"
        )
        need_probe = adaptive_wanted or should_attempt_gstreamer(rtsp_mode)
        profile = None
        if need_probe:
            profile = StreamProber.probe(self.url, camera_id=self.camera_id)
            if self._force_cpu_decode and profile.hw_decoder_requested:
                log.warning(
                    "[STREAM] HW decoder disabled for camera=%s after device errors; using CPU decode",
                    self.camera_id,
                )
                profile.hw_decoder_requested = None
                profile.hw_decoder_active = False
                profile.decoder = "cpu"

        # 1) GStreamer (Jetson NVDEC) — before FFmpeg subprocess
        if (
            profile is not None
            and should_attempt_gstreamer(rtsp_mode)
            and _codec_ok_for_gstreamer(profile.codec)
        ):
            gst = open_gstreamer_capture(profile)
            if gst is not None:
                self.cap, health = gst
                self._gstreamer_profile = profile
                self._ingest_backend = "gstreamer"
                _current_rtsp_backend = "gstreamer"
                _current_quality_label = f"{profile.target_width}x{profile.target_height}@gstreamer"
                _current_decoder_type = "nvv4l2decoder"
                _current_hw_decoder_requested = "nvv4l2decoder"
                _current_hw_decoder_active = True
                _current_stream_health = {**health, "backend": "gstreamer"}
                log.info(
                    "[STREAM] GStreamer NVDEC connected — native=%dx%d@%.2ffps codec=%s target=%s",
                    profile.native_width,
                    profile.native_height,
                    profile.native_fps,
                    profile.codec,
                    _current_quality_label,
                )
                wup = max(0, int(os.getenv("RTSP_GST_WARMUP_FRAMES", os.getenv("RTSP_WARMUP_FRAMES", "3"))))
                for _ in range(wup):
                    if self.cap is not None:
                        self.cap.read()
                return
            if rtsp_mode == "gstreamer":
                raise RuntimeError(
                    f"[STREAM] RTSP_BACKEND=gstreamer but failed to open pipeline for {self.url!r}"
                )
            log.warning(
                "[STREAM] GStreamer open failed for camera=%s; falling back to Adaptive FFmpeg / OpenCV",
                self.camera_id,
            )
        elif profile is not None and should_attempt_gstreamer(rtsp_mode) and not _codec_ok_for_gstreamer(profile.codec):
            log.info(
                "[STREAM] GStreamer skipped (codec=%s) — not H.264/H.265",
                profile.codec,
            )

        # 2) Adaptive FFmpeg
        if adaptive_wanted:
            if profile is None:
                profile = StreamProber.probe(self.url, camera_id=self.camera_id)
                if self._force_cpu_decode and profile.hw_decoder_requested:
                    log.warning(
                        "[STREAM] HW decoder disabled for camera=%s after device errors; using CPU decode",
                        self.camera_id,
                    )
                    profile.hw_decoder_requested = None
                    profile.hw_decoder_active = False
                    profile.decoder = "cpu"
            stream = AdaptiveStream(profile)
            if stream.open():
                self._adaptive_profile = profile
                self._adaptive = stream
                self._ingest_backend = "adaptive_ffmpeg"
                _current_rtsp_backend = "adaptive_ffmpeg"
                _current_quality_label = (
                    f"{profile.target_width}x{profile.target_height}@{profile.target_fps:.2f}fps"
                )
                _current_decoder_type = profile.decoder
                _current_hw_decoder_requested = profile.hw_decoder_requested
                _current_hw_decoder_active = profile.hw_decoder_active
                _current_stream_health = {**profile_health(profile), "backend": "adaptive_ffmpeg"}
                log.info(
                    "[STREAM] Adaptive FFmpeg connected — native=%dx%d@%.2ffps codec=%s target=%s decoder=%s",
                    profile.native_width,
                    profile.native_height,
                    profile.native_fps,
                    profile.codec,
                    _current_quality_label,
                    profile.decoder,
                )
                return

            log.warning("[STREAM] Adaptive FFmpeg unavailable; falling back to OpenCV VideoCapture")

        opts = apply_rtsp_ffmpeg_env()
        log.info(
            "[STREAM] Connecting (attempt %d): %s | ffmpeg_opts=%r",
            self._connect_count,
            self.url,
            opts,
        )
        self.cap = open_rtsp_videocapture(self.url)
        if not self.cap.isOpened():
            raise RuntimeError(f"[STREAM] Cannot open: {self.url}")
        self._ingest_backend = "opencv_ffmpeg"
        _current_rtsp_backend = "opencv_ffmpeg"
        wup = max(0, int(os.getenv("RTSP_WARMUP_FRAMES", "8")))
        okw = warmup_rtsp_capture(self.cap)
        w   = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 0.0
        _current_quality_label = QUALITY_LADDER[0]["label"] if QUALITY_LADDER else "live"
        _current_decoder_type = "cpu"
        _current_hw_decoder_requested = None
        _current_hw_decoder_active = False
        _current_stream_health = {
            "camera_id": self.camera_id,
            "codec": "opencv",
            "native": f"{w}x{h}@{round(fps, 2)}fps",
            "target": "FrameBus resize",
            "quality_tier": "opencv",
            "decoder": "cpu",
            "hw_decoder_requested": None,
            "hw_decoder_active": False,
            "healthy": True,
            "backend": "opencv_ffmpeg",
        }
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
        global _current_stream_health
        if self._ingest_backend == "gstreamer" and self.cap is not None:
            ret, frame = self.cap.read()
            if self._gstreamer_profile is not None:
                if not ret or frame is None:
                    self._gstreamer_profile.frames_dropped += 1
                    self._gstreamer_profile.healthy = False
                else:
                    self._gstreamer_profile.frames_received += 1
                    self._gstreamer_profile.healthy = True
                _current_stream_health = {
                    **profile_health(self._gstreamer_profile),
                    "backend": "gstreamer",
                    "decoder": "nvv4l2decoder",
                    "hw_decoder_requested": "nvv4l2decoder",
                    "hw_decoder_active": True,
                }
            return frame if ret else None
        if self._adaptive is not None:
            frame = self._adaptive.read_frame()
            if self._adaptive_profile is not None:
                if frame is None:
                    self._adaptive_profile.frames_dropped += 1
                    self._adaptive_profile.healthy = False
                h = {**profile_health(self._adaptive_profile), "backend": "adaptive_ffmpeg"}
                _current_stream_health = h
            return frame
        if self.cap is None:
            return None
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self) -> None:
        if self._adaptive is not None:
            self._adaptive.close()
            self._adaptive = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self._gstreamer_profile = None
        self._ingest_backend = None


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
    reader  = _RTSPReader(source, camera_id=camera_id) if is_rtsp else _VideoReader(source)

    consecutive_fails = 0
    reconnect_streak  = 0
    throttle = _FrameThrottle.from_env()
    adaptive_failover_after = max(1, _env_int("RTSP_ADAPTIVE_FAILOVER_AFTER", 4, minimum=1))

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
                        if getattr(reader, "_adaptive", None) is not None:
                            reader._adaptive.log_stderr_tail("reconnect_after_read_failures")
                            if reader._adaptive.should_disable_hw_decoder():
                                reader._force_cpu_decode = True
                                log.warning(
                                    "[STREAM] Camera %s: disabling HW decode due to ffmpeg device errors",
                                    reader.camera_id,
                                )
                            reader._adaptive_failure_cycles += 1
                            if (
                                not reader._disable_adaptive_ffmpeg
                                and reader._adaptive_failure_cycles >= adaptive_failover_after
                            ):
                                reader._disable_adaptive_ffmpeg = True
                                log.warning(
                                    "[STREAM] Camera %s: adaptive ffmpeg failed %d cycles; "
                                    "switching to OpenCV fallback",
                                    reader.camera_id,
                                    reader._adaptive_failure_cycles,
                                )
                        reader.release()
                        time.sleep(wait)
                        try:
                            reader.connect()
                            _rtsp_reconnect_count += 1
                            if is_rtsp and getattr(reader, "_adaptive_profile", None) is not None:
                                reader._adaptive_profile.reconnects = _rtsp_reconnect_count
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
            if is_rtsp and getattr(reader, "_adaptive", None) is not None:
                reader._adaptive_failure_cycles = 0
            if not throttle.should_yield():
                continue
            yield frame
    finally:
        reader.release()
