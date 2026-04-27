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
import shutil
import subprocess
import threading
import time
from typing import Optional

import cv2
from dotenv import load_dotenv

load_dotenv()

from logger.logger_config import Logger
log = Logger.get_logger(__name__)

USE_STREAM  = os.getenv("USE_STREAM",   "True").lower() in ("true", "1", "yes")
RTSP_URL    = os.getenv("CAMERA_1_URL", "")
INPUT_VIDEO = os.getenv("INPUT_VIDEO",  "./videos/sample.mp4")

_DEFAULT_FFMPEG_OPTIONS = {
    "rtsp_transport": "tcp",
    "timeout": "5000000",
    "reconnect": "1",
    "reconnect_delay_max": "5",
}
_CODEC_ALIASES = {
    "h264": "h264",
    "avc": "h264",
    "hevc": "h265",
    "h265": "h265",
}


def _merge_ffmpeg_capture_options(existing: str = "") -> str:
    """Merge RTSP defaults without discarding Docker/runtime overrides."""
    merged = dict(_DEFAULT_FFMPEG_OPTIONS)
    for part in (existing or "").split("|"):
        if not part:
            continue
        key, sep, value = part.partition(";")
        if sep and key:
            merged[key] = value
    return "|".join(f"{key};{value}" for key, value in merged.items())


def _opencv_has_gstreamer() -> bool:
    try:
        return "GStreamer:                   YES" in cv2.getBuildInformation()
    except Exception:
        return False


def _select_stream_backend() -> str:
    backend = os.getenv("STREAM_BACKEND", "auto").strip().lower()
    if backend in ("ffmpeg", "gstreamer"):
        return backend
    if backend != "auto":
        log.warning(f"[STREAM] Unknown STREAM_BACKEND={backend!r}; using auto")
    return "gstreamer" if _opencv_has_gstreamer() else "ffmpeg"


def _detect_rtsp_codec(url: str) -> str:
    codec_override = os.getenv("STREAM_CODEC", "auto").strip().lower()
    if codec_override in _CODEC_ALIASES:
        return _CODEC_ALIASES[codec_override]
    if codec_override != "auto":
        log.warning(f"[STREAM] Unknown STREAM_CODEC={codec_override!r}; falling back to auto")

    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return os.getenv("STREAM_CODEC_FALLBACK", "h264").strip().lower()

    timeout = float(os.getenv("STREAM_FFPROBE_TIMEOUT_SEC", "5"))
    cmd = [
        ffprobe,
        "-v", "error",
        "-rtsp_transport", os.getenv("STREAM_RTSP_TRANSPORT", "tcp"),
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name",
        "-of", "default=noprint_wrappers=1:nokey=1",
        url,
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        log.warning(f"[STREAM] ffprobe timed out after {timeout:.1f}s; using fallback codec")
        return os.getenv("STREAM_CODEC_FALLBACK", "h264").strip().lower()
    except Exception as exc:
        log.warning(f"[STREAM] ffprobe failed ({exc}); using fallback codec")
        return os.getenv("STREAM_CODEC_FALLBACK", "h264").strip().lower()

    codec = (result.stdout or "").strip().splitlines()
    if codec:
        return _CODEC_ALIASES.get(codec[0].lower(), os.getenv("STREAM_CODEC_FALLBACK", "h264").strip().lower())
    return os.getenv("STREAM_CODEC_FALLBACK", "h264").strip().lower()


def _build_gstreamer_pipeline(url: str, codec: str) -> str:
    latency = int(os.getenv("STREAM_GST_LATENCY_MS", "100"))
    transport = os.getenv("STREAM_RTSP_TRANSPORT", "tcp").strip().lower()
    protocols = "tcp" if transport == "tcp" else "udp"
    decoder = os.getenv("STREAM_GST_DECODER", "nvv4l2decoder")
    depay_parse = {
        "h264": "application/x-rtp,media=video,encoding-name=H264 ! rtph264depay ! h264parse",
        "h265": "application/x-rtp,media=video,encoding-name=H265 ! rtph265depay ! h265parse",
    }.get(codec, "decodebin")

    if depay_parse == "decodebin":
        decode_chain = "decodebin"
    else:
        decode_chain = f"{depay_parse} ! {decoder}"

    return (
        f"rtspsrc location=\"{url}\" protocols={protocols} latency={latency} "
        "drop-on-latency=true ! "
        f"{decode_chain} ! "
        "nvvidconv ! video/x-raw,format=BGRx ! "
        "videoconvert ! video/x-raw,format=BGR ! "
        "appsink drop=true max-buffers=1 sync=false"
    )


class _RTSPReader:
    def __init__(self, url: str):
        self.url = url
        self.cap: Optional[cv2.VideoCapture] = None
        self._requested_backend = os.getenv("STREAM_BACKEND", "auto").strip().lower()
        self.backend = _select_stream_backend()
        self._allow_backend_fallback = self._requested_backend not in ("ffmpeg", "gstreamer")
        self.codec = "unknown"
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._latest_frame = None
        self._last_frame_at = 0.0
        self._last_error: Optional[str] = None
        self._reconnects = 0

    def connect(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._reader_loop, name="rtsp-reader", daemon=True)
        self._thread.start()

    def _open_capture(self):
        self.codec = _detect_rtsp_codec(self.url) if self.backend == "gstreamer" else "auto"
        if self.backend == "gstreamer":
            source = _build_gstreamer_pipeline(self.url, self.codec)
            api = cv2.CAP_GSTREAMER
        else:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = _merge_ffmpeg_capture_options(
                os.getenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", "")
            )
            source = self.url
            api = cv2.CAP_FFMPEG

        log.info(f"[STREAM] Connecting via {self.backend} codec={self.codec}: {self.url}")
        cap = cv2.VideoCapture(source, api)
        self.cap = cap
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError(f"[STREAM] Cannot open: {self.url}")
        w   = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        log.info(f"[STREAM] Connected — {w}x{h} @ {fps:.1f}fps")

    def _reader_loop(self):
        backoff = float(os.getenv("STREAM_RECONNECT_INITIAL_SEC", "1"))
        max_backoff = float(os.getenv("STREAM_RECONNECT_MAX_SEC", "10"))
        while not self._stop_event.is_set():
            try:
                self._open_capture()
                backoff = float(os.getenv("STREAM_RECONNECT_INITIAL_SEC", "1"))
                while not self._stop_event.is_set():
                    ret, frame = self.cap.read()
                    if not ret or frame is None:
                        self._last_error = "read failed"
                        break
                    with self._lock:
                        self._latest_frame = frame.copy()
                        self._last_frame_at = time.time()
                        self._last_error = None
            except Exception as exc:
                self._last_error = str(exc)
                log.warning(f"[STREAM] Reader error: {exc}")
                if self.backend == "gstreamer" and self._allow_backend_fallback:
                    log.warning("[STREAM] Falling back to FFmpeg backend after GStreamer open/read failure")
                    self.backend = "ffmpeg"
            finally:
                self._release_capture()

            if self._stop_event.is_set():
                break
            self._reconnects += 1
            log.info(f"[STREAM] Reconnecting in {backoff:.1f}s (attempt {self._reconnects})")
            self._stop_event.wait(backoff)
            backoff = min(max_backoff, backoff * 2)

    def read_frame(self):
        with self._lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

    def _release_capture(self):
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def release(self):
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)
        self._thread = None
        self._release_capture()


class _VideoReader:
    def __init__(self, path: str):
        self.path = path
        self.cap  = None

    def connect(self):
        log.info(f"[VIDEO] Opening: {self.path}")
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise RuntimeError(f"[VIDEO] Cannot open: {self.path}")

    def read_frame(self):
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        if self.cap:
            self.cap.release()


def capture_preview_frame(
    source: str,
    *,
    max_reads: int = 15,
    settle_after_reads: int = 5,
):
    """
    Grab one BGR frame using the same RTSP / file pipeline as ``frames()``
    (GStreamer or FFmpeg, codec detection, etc.).

    Discards the first ``settle_after_reads - 1`` yielded frames (decoder warm-up),
    then returns a copy of the next frame, or ``None`` if not enough frames arrive
    within ``max_reads`` yields.
    """
    if settle_after_reads < 1:
        settle_after_reads = 1
    if max_reads < settle_after_reads:
        max_reads = settle_after_reads

    gen = frames(source)
    try:
        chosen = None
        for i in range(max_reads):
            try:
                frame = next(gen)
            except StopIteration:
                break
            if i + 1 >= settle_after_reads:
                chosen = frame
                break
        return None if chosen is None else chosen.copy()
    finally:
        gen.close()


def frames(source: str = None):
    """
    Generator yielding BGR numpy frames.

    Args:
        source: RTSP URL, video file path, or None (uses .env defaults)
    """
    if source is None:
        source = RTSP_URL if USE_STREAM else INPUT_VIDEO

    is_rtsp = source.startswith("rtsp://")
    reader  = _RTSPReader(source) if is_rtsp else _VideoReader(source)

    consecutive_fails = 0
    max_fails         = int(os.getenv("STREAM_READ_MAX_FAILS", "50"))
    sleep_on_fail     = float(os.getenv("STREAM_READ_SLEEP_SEC", "0.02"))

    reader.connect()
    try:
        while True:
            frame = reader.read_frame()
            if frame is None:
                consecutive_fails += 1
                if not is_rtsp and consecutive_fails >= max_fails:
                    break  # end of video file
                time.sleep(sleep_on_fail)
                continue
            consecutive_fails = 0
            yield frame
    finally:
        reader.release()