"""
Adaptive RTSP stream adapter.

Uses ffprobe + ffmpeg rawvideo output to normalize mixed camera feeds before
the detection pipeline sees them. It keeps OpenCV out of the hot decode/resize
path for RTSP sources while preserving a small, generator-friendly API.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from typing import BinaryIO, Optional

import numpy as np

logger = logging.getLogger("shared_logger.StreamAdapter")


class StreamQuality(Enum):
    ULTRA = "ultra"
    HIGH = "high"
    MED = "med"
    LOW = "low"


@dataclass
class CameraProfile:
    camera_id: str
    url: str
    native_width: int = 0
    native_height: int = 0
    native_fps: float = 0.0
    codec: str = "unknown"
    has_audio: bool = False
    quality: StreamQuality = StreamQuality.HIGH
    target_width: int = 480
    target_height: int = 360
    target_fps: float = 8.0
    frames_received: int = 0
    frames_dropped: int = 0
    reconnects: int = 0
    last_frame_ts: float = field(default_factory=time.time)
    healthy: bool = True
    decoder: str = "cpu"
    hw_decoder_requested: Optional[str] = None
    hw_decoder_active: bool = False


def _env_flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).lower() in ("true", "1", "yes", "on")


def _env_float(name: str, default: float, minimum: float = 0.0) -> float:
    try:
        return max(minimum, float(os.getenv(name, str(default))))
    except ValueError:
        return max(minimum, default)


def _is_jetson() -> bool:
    if os.path.exists("/etc/nv_tegra_release"):
        return True
    try:
        with open("/proc/device-tree/model", "r", encoding="utf-8") as fh:
            model = fh.read().lower()
            return "nvidia" in model and "jetson" in model
    except OSError:
        return False


class StreamProber:
    JETSON_HW_DECODERS = {
        "h264": "h264_v4l2m2m",
        "hevc": "hevc_v4l2m2m",
        "h265": "hevc_v4l2m2m",
    }

    QUALITY_MAP = (
        (3840, StreamQuality.ULTRA),
        (1920, StreamQuality.HIGH),
        (1280, StreamQuality.MED),
        (0, StreamQuality.LOW),
    )

    TARGET_SIZE = {
        StreamQuality.ULTRA: (640, 480, 5.0),
        StreamQuality.HIGH: (480, 360, 8.0),
        StreamQuality.MED: (384, 288, 10.0),
        StreamQuality.LOW: (320, 240, 15.0),
    }

    @classmethod
    def probe(cls, url: str, camera_id: str | None = None, timeout: int | None = None) -> CameraProfile:
        profile = CameraProfile(camera_id=camera_id or "camera", url=url)
        timeout = timeout or int(os.getenv("STREAM_PROBE_TIMEOUT_SEC", "8"))
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-rtsp_transport",
            os.getenv("RTSP_TRANSPORT", "tcp"),
            "-analyzeduration",
            os.getenv("STREAM_PROBE_ANALYZE_DURATION", "3000000"),
            "-probesize",
            os.getenv("STREAM_PROBE_SIZE", "1000000"),
            "-print_format",
            "json",
            "-show_streams",
            url,
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip() or f"ffprobe exited {result.returncode}")
            data = json.loads(result.stdout or "{}")
            cls._apply_probe_data(profile, data)
        except FileNotFoundError:
            logger.warning("[%s] ffprobe not found; using safe stream defaults", profile.camera_id)
        except (json.JSONDecodeError, RuntimeError, subprocess.TimeoutExpired, OSError) as exc:
            logger.warning("[%s] ffprobe failed (%s); using safe stream defaults", profile.camera_id, exc)

        cls._derive_targets(profile)
        return profile

    @classmethod
    def _apply_probe_data(cls, profile: CameraProfile, data: dict) -> None:
        for stream in data.get("streams", []):
            codec_type = stream.get("codec_type")
            if codec_type == "video":
                profile.native_width = int(stream.get("width") or 0)
                profile.native_height = int(stream.get("height") or 0)
                profile.codec = str(stream.get("codec_name") or "unknown").lower()
                profile.native_fps = cls._parse_fps(
                    stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1"
                )
            elif codec_type == "audio":
                profile.has_audio = True

    @staticmethod
    def _parse_fps(raw: str) -> float:
        try:
            value = float(Fraction(str(raw)))
        except (ValueError, ZeroDivisionError):
            return 0.0
        return round(value, 2) if value > 0 else 0.0

    @classmethod
    def _derive_targets(cls, profile: CameraProfile) -> None:
        for min_width, quality in cls.QUALITY_MAP:
            if profile.native_width >= min_width:
                profile.quality = quality
                break

        width, height, fps = cls.TARGET_SIZE[profile.quality]
        profile.target_width = int(os.getenv("STREAM_ADAPTER_WIDTH", str(width)))
        profile.target_height = int(os.getenv("STREAM_ADAPTER_HEIGHT", str(height)))
        target_fps = _env_float("STREAM_ADAPTER_TARGET_FPS", fps, minimum=0.1)
        profile.target_fps = min(profile.native_fps or target_fps, target_fps)

        decoder = cls.get_decoder(profile.codec)
        profile.hw_decoder_requested = decoder
        profile.decoder = decoder or "cpu"
        profile.hw_decoder_active = bool(decoder)

        logger.info(
            "[%s] Probed stream: %dx%d @ %.2ffps codec=%s -> %dx%d @ %.2ffps tier=%s decoder=%s",
            profile.camera_id,
            profile.native_width,
            profile.native_height,
            profile.native_fps,
            profile.codec,
            profile.target_width,
            profile.target_height,
            profile.target_fps,
            profile.quality.value,
            profile.decoder,
        )

    @classmethod
    def get_decoder(cls, codec: str) -> Optional[str]:
        mode = os.getenv("RTSP_ENABLE_GPU_DECODE", "auto").lower()
        if mode in ("false", "0", "no", "off"):
            return None
        if mode == "auto" and not _is_jetson():
            return None

        c = (codec or "").lower()
        default = cls.JETSON_HW_DECODERS.get(c)
        if default is None:
            return None

        if c in ("hevc", "h265"):
            o = os.getenv("RTSP_JETSON_HEVC_DECODER", "").strip()
            if o:
                return o
        elif c == "h264":
            o = os.getenv("RTSP_JETSON_H264_DECODER", "").strip()
            if o:
                return o

        legacy = os.getenv("RTSP_HWDECODER", "").strip()
        if legacy:
            return legacy
        return default


class AdaptiveStream:
    BASE_INPUT_OPTS = (
        "-rtsp_transport",
        os.getenv("RTSP_TRANSPORT", "tcp"),
        "-fflags",
        "genpts+discardcorrupt",
        "-max_delay",
        os.getenv("RTSP_MAX_DELAY_US", "3000000"),
        "-reorder_queue_size",
        os.getenv("RTSP_REORDER_QUEUE_SIZE", "512"),
    )

    def __init__(self, profile: CameraProfile):
        self.profile = profile
        self._proc: subprocess.Popen | None = None
        self._running = False
        self._stderr_tail = bytearray()
        self._stderr_thread: threading.Thread | None = None
        self._stderr_max = max(1024, int(os.getenv("STREAM_ADAPTER_FFMPEG_STDERR_MAX", "8192")))

    def _build_ffmpeg_cmd(self) -> list[str]:
        p = self.profile
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", os.getenv("STREAM_ADAPTER_LOGLEVEL", "error")]
        if p.hw_decoder_requested:
            cmd += ["-c:v", p.hw_decoder_requested]
        cmd += list(self.BASE_INPUT_OPTS)
        cmd += ["-i", p.url]
        vf = f"fps={p.target_fps:.2f},scale={p.target_width}:{p.target_height}:flags=fast_bilinear"
        cmd += [
            "-vf",
            vf,
            "-an",
            "-sn",
            "-dn",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "pipe:1",
        ]
        return cmd

    def _stderr_drainer(self, fh: BinaryIO) -> None:
        try:
            while True:
                chunk = fh.read(4096)
                if not chunk:
                    break
                self._stderr_tail.extend(chunk)
                excess = len(self._stderr_tail) - self._stderr_max
                if excess > 0:
                    del self._stderr_tail[:excess]
        except Exception:
            pass
        finally:
            try:
                fh.close()
            except Exception:
                pass

    def log_stderr_tail(self, reason: str) -> None:
        if not self._stderr_tail:
            return
        text = self._stderr_tail.decode("utf-8", errors="replace").strip()
        if text:
            logger.warning("[%s] ffmpeg stderr tail (%s): %s", self.profile.camera_id, reason, text)

    def open(self) -> bool:
        self._stderr_tail.clear()
        self._stderr_thread = None
        cmd = self._build_ffmpeg_cmd()
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        except FileNotFoundError:
            logger.warning("[%s] ffmpeg not found; adaptive stream unavailable", self.profile.camera_id)
            return False
        except OSError as exc:
            logger.warning("[%s] ffmpeg start failed: %s", self.profile.camera_id, exc)
            return False

        if self._proc.stderr is not None:
            self._stderr_thread = threading.Thread(
                target=self._stderr_drainer,
                args=(self._proc.stderr,),
                daemon=True,
                name=f"ffmpeg-stderr-{self.profile.camera_id}",
            )
            self._stderr_thread.start()

        self._running = True
        logger.info("[%s] ffmpeg started pid=%s decoder=%s", self.profile.camera_id, self._proc.pid, self.profile.decoder)
        return True

    def read_frame(self) -> np.ndarray | None:
        if not self._running or self._proc is None or self._proc.stdout is None:
            return None
        if self._proc.poll() is not None:
            return None

        p = self.profile
        frame_bytes = p.target_width * p.target_height * 3
        try:
            raw = self._proc.stdout.read(frame_bytes)
        except OSError:
            return None
        if len(raw) != frame_bytes:
            return None

        p.frames_received += 1
        p.last_frame_ts = time.time()
        p.healthy = True
        return np.frombuffer(raw, dtype=np.uint8).reshape((p.target_height, p.target_width, 3))

    def close(self) -> None:
        self._running = False
        if self._proc is None:
            return
        proc = self._proc
        self._proc = None
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3)
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=2.0)
            self._stderr_thread = None
        logger.info("[%s] ffmpeg process closed", self.profile.camera_id)


def profile_health(profile: CameraProfile) -> dict:
    return {
        "camera_id": profile.camera_id,
        "codec": profile.codec,
        "native": f"{profile.native_width}x{profile.native_height}@{profile.native_fps}fps",
        "target": f"{profile.target_width}x{profile.target_height}@{profile.target_fps}fps",
        "quality_tier": profile.quality.value,
        "decoder": profile.decoder,
        "hw_decoder_requested": profile.hw_decoder_requested,
        "hw_decoder_active": profile.hw_decoder_active,
        "frames_received": profile.frames_received,
        "frames_dropped": profile.frames_dropped,
        "reconnects": profile.reconnects,
        "healthy": profile.healthy,
        "last_frame_age_s": round(time.time() - profile.last_frame_ts, 1),
    }
