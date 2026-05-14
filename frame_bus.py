"""
frame_bus.py
------------
FrameBus — runs inside each camera process.

Captures frames, runs YOLO BoT-SORT tracking, then fans out
{frame + tracked detections} to each registered task queue.

Additionally saves the best crop per tracked person (progressive overwrite)
and emits lightweight messages to an embedding_queue for async embedding
extraction by the EmbeddingWorker.

Track IDs are assigned here and carried on each Detection object,
so task workers never need to run their own detector or tracker.
"""

import os
import sys

# Ensure repo root is on sys.path in worker processes (multiprocessing / some uvicorn setups
# omit PYTHONPATH). Ultralytics BoT-SORT imports `lap`; we ship a compatible `lap/` package.
_repo_root = os.path.dirname(os.path.abspath(__file__))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

_DEFAULT_TRACKER_YAML = os.path.join(_repo_root, "cfg", "trackers", "botsort_stable.yaml")

import json
import time
import base64
import hashlib
import queue as _queue
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from logger.logger_config import Logger
from utils import resize, save_frame
from services.detector import Detection
from resilience.circuit_breaker import CircuitBreaker
from resilience.sequencer import next_seq

log = Logger.get_logger(__name__)

try:
    import redis as _redis_lib
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False

# Minimum confidence / variance to emit crop to embedding worker
_EMBED_CONF_THRESHOLD = float(os.getenv("EMBED_CONF_THRESHOLD", "0.45"))
_EMBED_MIN_VARIANCE = float(os.getenv("FRAME_MIN_VARIANCE", "8.0"))
# Minimum Laplacian variance for sharpness gate — rejects motion-blurred crops
# that would corrupt the ReID gallery. Lower = more permissive; 15.0 is a safe
# default for 480p edge feeds. Set EMBED_MIN_SHARPNESS=0 to disable.
_EMBED_MIN_SHARPNESS = float(os.getenv("EMBED_MIN_SHARPNESS", "15.0"))

# Live JPEG / preview scale tiers (corrupt rate + optional WS backpressure).
LIVE_QUALITY_LADDER = (
    (85, 1.0, "high"),
    (75, 0.75, "medium"),
    (60, 0.5, "low"),
    (50, 0.4, "minimal"),
)


class CorruptionDetector:
    """Fast multi-check gate for HEVC decode glitches before JPEG / YOLO."""

    def __init__(self, threshold_ratio: float = 0.35):
        self.threshold = threshold_ratio
        self._last_good_frame = None
        self._corrupt_streak = 0
        self._consecutive_corrupt_all = 0

    @property
    def consecutive_corrupt_all(self) -> int:
        return self._consecutive_corrupt_all

    def is_corrupted(self, frame: np.ndarray) -> bool:
        if frame is None or frame.size == 0:
            return True
        h, w = frame.shape[:2]
        if h < 16 or w < 16:
            return True

        mean_lum = float(frame.mean())
        if mean_lum < 5.0 or mean_lum > 250.0:
            return True

        if frame.ndim == 3 and frame.shape[2] >= 3:
            samples = [
                frame[h // 8, w // 8],
                frame[h // 8, 7 * w // 8],
                frame[7 * h // 8, w // 8],
                frame[7 * h // 8, 7 * w // 8],
                frame[h // 2, w // 2],
            ]
            for px in samples:
                b, g, r = int(px[0]), int(px[1]), int(px[2])
                if g > 180 and g > r * 3 and g > b * 3:
                    return True
                if r > 150 and b > 150 and g < 50:
                    return True

        if self._last_good_frame is not None and self._last_good_frame.shape == frame.shape:
            cy, cx = h // 2, w // 2
            dy, dx = max(1, h // 10), max(1, w // 10)
            patch_curr = frame[cy - dy : cy + dy, cx - dx : cx + dx]
            patch_prev = self._last_good_frame[cy - dy : cy + dy, cx - dx : cx + dx]
            diff = np.abs(patch_curr.astype(np.int16) - patch_prev.astype(np.int16)).mean()
            if diff > 120.0:
                self._corrupt_streak += 1
                if self._corrupt_streak >= 2:
                    return True
            else:
                self._corrupt_streak = 0

        return False

    def process(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], bool]:
        """
        Returns (frame to use, was_corrupt_input).
        On corrupt input returns last good copy (or None if none yet) and was_corrupt_input True.
        """
        if self.is_corrupted(frame):
            self._consecutive_corrupt_all += 1
            return self._last_good_frame, True
        self._consecutive_corrupt_all = 0
        self._last_good_frame = frame
        self._corrupt_streak = 0
        return frame, False


class PublishCircuitBreaker:
    """Redis publish backpressure: buffer a short burst while circuit is open."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 10.0):
        self.failures = 0
        self.threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.last_failure_time = 0.0
        self.state = "closed"
        self._buffer: deque[str] = deque(maxlen=30)

    def record_failure(self) -> None:
        self.failures += 1
        self.last_failure_time = time.time()
        if self.failures >= self.threshold:
            self.state = "open"
            log.warning(
                "Redis publish circuit OPEN — buffering frames locally (max %d)",
                self._buffer.maxlen,
            )

    def record_success(self) -> None:
        self.failures = 0
        self.state = "closed"

    def should_publish(self) -> bool:
        if self.state == "closed":
            return True
        if self.state == "open":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half-open"
                return True
            return False
        return True

    def buffer_envelope(self, envelope: str) -> None:
        self._buffer.append(envelope)

    def drain_buffer(self) -> list[str]:
        out = list(self._buffer)
        self._buffer.clear()
        return out


class FrameBus:
    def __init__(
        self,
        camera_id       : str,
        rtsp_url        : str,
        shared_state,
        stop_event,
        task_queues     : Dict[str, object],  # {task_id: Queue}
        embedding_queue = None,                # Queue for EmbeddingWorker
        frame_seq_counter: Any = None,         # multiprocessing.Value('Q') for live frames
        frame_seq_lock: Any = None,
        bus_fatal_event: Any = None,           # multiprocessing.Event — set on fatal error
        live_overlay: Optional[Dict[str, Any]] = None,  # cross-line + cashier zones (see utils/live_stream_overlay.py)
    ):
        self.camera_id       = camera_id
        self.rtsp_url        = rtsp_url
        self.shared_state    = shared_state
        self.stop_event      = stop_event
        self.task_queues     = task_queues
        self.embedding_queue = embedding_queue
        self._frame_seq_counter = frame_seq_counter
        self._frame_seq_lock = frame_seq_lock
        self._bus_fatal_event = bus_fatal_event
        self._live_overlay = live_overlay or {}
        _geom_env = os.getenv("LIVE_STREAM_GEOMETRY_OVERLAY", "true").lower()
        self._live_stream_geometry_enabled = _geom_env in ("true", "1", "yes")
        self._live_stream_geometry_active = self._live_stream_geometry_enabled and (
            bool(self._live_overlay.get("cross_lines"))
            or bool(self._live_overlay.get("cashier_zones"))
        )

        # Default off: per-frame disk writes fill storage quickly; enable explicitly for debugging.
        self.save_output = os.getenv("SAVE_OUTPUT", "false").lower() in ("true", "1", "yes")
        self.out_dir     = os.path.join(os.getenv("OUTPUT_DIR", "./outputs"), camera_id)
        self.width       = int(os.getenv("WIDTH",  "1280"))
        self.height      = int(os.getenv("HEIGHT", "0"))
        self._padding    = int(os.getenv("REID_PADDING", "10"))

        model_path   = os.getenv("YOLO_MODEL", "yolov8n.pt")
        conf         = float(os.getenv("CONF_THRESHOLD", "0.45"))
        # NMS IoU: higher → suppress fewer overlapping boxes (better for crowded groups).
        iou          = float(os.getenv("IOU_THRESHOLD", "0.55"))
        _device_raw  = os.getenv("DEVICE", "0")
        device       = int(_device_raw) if _device_raw.isdigit() else _device_raw
        _classes_raw = os.getenv("FILTER_CLASSES", "")
        classes      = [int(c.strip()) for c in _classes_raw.split(",") if c.strip()] or None

        try:
            self._max_det = max(1, int(os.getenv("YOLO_MAX_DET", "300")))
        except ValueError:
            self._max_det = 300
        _imgsz_raw = os.getenv("YOLO_IMGSZ", "").strip()
        self._imgsz: Optional[int] = int(_imgsz_raw) if _imgsz_raw.isdigit() else None

        self._model   = YOLO(model_path, task="detect")
        self._conf    = conf
        self._iou     = iou
        self._device  = device
        self._classes = classes
        self._names   = self._model.names

        from utils.ml_backend import require_gpu_device_if_configured, resolve_ultralytics_device

        require_gpu_device_if_configured(
            resolve_ultralytics_device(), "FrameBus"
        )

        # ── Best-crop-per-track state ─────────────────────────────────────
        # { track_id: {"best_area": int, "last_frame": int} }
        self._track_state: Dict[int, Dict] = {}
        self._gallery_dir = os.getenv("GALLERY_DIR", "/local/storage/gallery")
        self._crop_dir    = os.path.join(self._gallery_dir, "crops", camera_id)
        os.makedirs(self._crop_dir, exist_ok=True)

        # ── Redis live-stream publisher ────────────────────────────────────
        # Publishes annotated JPEG frames to channel  live:frame:{camera_id}
        # so the FastAPI WebSocket endpoint can fan them out to browser clients.
        self._redis: Optional[object] = None
        self._redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self._redis_connect_retries = max(1, int(os.getenv("REDIS_CONNECT_RETRIES", "5")))
        self._redis_health_interval = max(1, int(os.getenv("REDIS_HEALTH_INTERVAL_FRAMES", "300")))
        self._redis_breaker = CircuitBreaker(
            name=f"framebus_redis:{camera_id}",
            failure_threshold=max(1, int(os.getenv("REDIS_CIRCUIT_FAILURES", "5"))),
            reset_timeout_sec=max(1.0, float(os.getenv("REDIS_CIRCUIT_RESET_SEC", "30"))),
        )
        if _REDIS_AVAILABLE:
            self._redis = self._connect_redis_with_retry()
            if self._redis is not None:
                log.info(
                    "[%s] FrameBus: Redis connected (%s)", camera_id, self._redis_url
                )
            else:
                log.warning(
                    "[%s] FrameBus: Redis unavailable — live stream disabled (will retry)",
                    camera_id,
                )

        if self._redis is not None:
            try:
                self._publish_quality_tier_redis()
            except Exception:
                pass

        # Publish every Nth frame to hit the target REDIS_LIVE_FPS.
        # We don't know the actual camera FPS at init time, so we start with
        # a conservative default and recalculate after the first FPS measurement.
        _target_fps        = float(os.getenv("REDIS_LIVE_FPS", "13"))
        self._publish_every = max(1, round(25.0 / _target_fps))  # assume 25 FPS until measured
        self._target_fps    = _target_fps

        self._frames_dropped = 0
        self._frames_passed  = 0
        self._embed_skipped  = 0
        self._embed_passed   = 0

        _tracker_env = os.getenv("TRACKER_YAML", "").strip()
        self._tracker_yaml = _tracker_env or (
            _DEFAULT_TRACKER_YAML if os.path.isfile(_DEFAULT_TRACKER_YAML) else "botsort.yaml"
        )

        # Wall-clock spacing between yielded frames (EMA) — jitter / stall indicator for metrics.
        self._metrics_last_wall_t: Optional[float] = None
        self._metrics_inter_frame_ema_ms = 0.0

        # Per-task queue backpressure (put_nowait failures) + throttled warnings
        self._task_queue_drops_by_task: Dict[str, int] = {
            str(k): 0 for k in self.task_queues
        }
        self._queue_warn_last: Dict[str, float] = {}
        self._queue_warn_interval = float(
            os.getenv("FRAMEBUS_QUEUE_WARN_INTERVAL_SEC", "5.0")
        )
        self._task_queue_maxsize = max(1, int(os.getenv("TASK_QUEUE_MAXSIZE", "256")))
        self._task_queue_coalesce = os.getenv("TASK_QUEUE_COALESCE", "true").lower() in (
            "true",
            "1",
            "yes",
        )
        self._task_queue_coalesce_threshold = self._env_float_clamped(
            "TASK_QUEUE_COALESCE_THRESHOLD", 0.8, 0.0, 1.0
        )
        self._include_frame_ndarray = os.getenv(
            "TASK_QUEUE_INCLUDE_FRAME", "false"
        ).lower() in ("true", "1", "yes")
        self._task_queue_coalesced_by_task: Dict[str, int] = {
            str(k): 0 for k in self.task_queues
        }

        # Default opencv: lighter than ultralytics plot(); set LIVE_ANNOTATION_MODE=ultralytics for rich labels.
        _lam = os.getenv("LIVE_ANNOTATION_MODE", "opencv").lower().strip()
        self._live_annotation_mode: str = (
            _lam if _lam in ("ultralytics", "opencv", "none") else "opencv"
        )

        try:
            self._task_jpeg_quality = max(
                1, min(100, int(os.getenv("TASK_QUEUE_JPEG_QUALITY", "85")))
            )
        except ValueError:
            self._task_jpeg_quality = 85
        try:
            self._live_jpeg_quality = max(
                1, min(100, int(os.getenv("LIVE_JPEG_QUALITY", "85")))
            )
        except ValueError:
            self._live_jpeg_quality = 85

        self._corruption = CorruptionDetector(
            threshold_ratio=float(os.getenv("CORRUPTION_DETECTOR_THRESHOLD", "0.35"))
        )
        self._integrity_events: deque = deque(maxlen=512)
        self._last_idr_signal_wall = 0.0
        self._last_keyframe_only_signal_wall = 0.0
        self._live_quality_tier = 0
        self._live_scale = 1.0
        self._above_bad_since: Optional[float] = None
        self._below_good_since: Optional[float] = None
        self._publish_frame_breaker = PublishCircuitBreaker(
            failure_threshold=max(1, int(os.getenv("REDIS_PUBLISH_CB_FAILURES", "5"))),
            recovery_timeout=float(os.getenv("REDIS_PUBLISH_CB_RECOVERY_SEC", "10.0")),
        )
        best_i = 0
        best_d = 999
        for i, row in enumerate(LIVE_QUALITY_LADDER):
            d = abs(int(row[0]) - self._live_jpeg_quality)
            if d < best_d:
                best_d = d
                best_i = i
        self._live_quality_tier = best_i
        _r0 = LIVE_QUALITY_LADDER[self._live_quality_tier]
        self._live_jpeg_quality = int(_r0[0])
        self._live_scale = float(_r0[1])

        self._state_update_every_n = max(
            1, int(os.getenv("STATE_UPDATE_INTERVAL", "10"))
        )
        try:
            self._state_update_min_sec = max(
                0.1, float(os.getenv("STATE_UPDATE_MIN_SEC", "1.0"))
            )
        except ValueError:
            self._state_update_min_sec = 1.0

        self._perf_log_every = max(1, int(os.getenv("PERF_LOG_INTERVAL", "300")))
        self._need_draw_warn_interval = max(
            1.0, float(os.getenv("FRAMEBUS_NEED_DRAW_WARN_SEC", "60.0"))
        )
        self._last_need_draw_warn_wall = 0.0

        self._last_live_publish_seq: int = 0
        self._last_live_frame_had_boxes: bool = False

        self._annotation_debug_every = max(
            1, int(os.getenv("ANNOTATION_DEBUG_LOG_INTERVAL", "1"))
        )

        # Optional MP4: same annotated BGR as Redis/WebSocket live preview (see _annotate_for_stream).
        self._save_annotated_video = os.getenv("SAVE_ANNOTATED_VIDEO", "false").lower() in (
            "true",
            "1",
            "yes",
        )
        self._annotated_video_path_raw = os.getenv("SAVE_ANNOTATED_VIDEO_PATH", "").strip()
        try:
            self._annotated_video_fps_env = float(os.getenv("SAVE_ANNOTATED_VIDEO_FPS", "0") or 0.0)
        except ValueError:
            self._annotated_video_fps_env = 0.0
        self._annotated_video_codec = os.getenv("SAVE_ANNOTATED_VIDEO_CODEC", "mp4v").strip().lower()
        self._annotated_video_writer: Optional[Any] = None
        self._annotated_video_open_path: Optional[str] = None
        self._annotated_video_failed = False
        # When true, append only on frames that would be published to Redis (same cadence as WS viewers).
        # If Redis is unavailable, every processed frame is written so local/test runs still get a file.
        self._annotated_video_match_ws = os.getenv(
            "SAVE_ANNOTATED_VIDEO_MATCH_WS", "false"
        ).lower() in ("true", "1", "yes")

    def _resolve_annotated_video_fps(self, fps_measured: float) -> float:
        if self._annotated_video_fps_env > 0.0:
            return max(1.0, min(self._annotated_video_fps_env, 120.0))
        try:
            stf = float(os.getenv("STREAM_TARGET_FPS", "0") or 0.0)
        except ValueError:
            stf = 0.0
        if stf > 0.0:
            return max(1.0, min(stf, 120.0))
        if fps_measured > 0.5:
            return max(1.0, min(fps_measured, 120.0))
        return 25.0

    def _annotated_video_fourcc(self) -> int:
        c = (self._annotated_video_codec or "mp4v").strip()
        if len(c) == 4:
            return cv2.VideoWriter_fourcc(*c)
        return cv2.VideoWriter_fourcc(*"mp4v")

    def _ensure_annotated_video_writer(self, frame_bgr: Any, fps_measured: float) -> None:
        if (
            not self._save_annotated_video
            or self._annotated_video_failed
            or self._annotated_video_writer is not None
        ):
            return
        h, w = frame_bgr.shape[:2]
        if h <= 0 or w <= 0:
            return
        if self._annotated_video_path_raw:
            out_mp4 = self._annotated_video_path_raw
            if not os.path.isabs(out_mp4):
                out_mp4 = os.path.join(_repo_root, out_mp4)
        else:
            os.makedirs(self.out_dir, exist_ok=True)
            out_mp4 = os.path.join(self.out_dir, "annotated_stream.mp4")
        parent = os.path.dirname(os.path.abspath(out_mp4))
        if parent:
            os.makedirs(parent, exist_ok=True)
        fps = self._resolve_annotated_video_fps(fps_measured)
        fourcc = self._annotated_video_fourcc()
        writer = cv2.VideoWriter(out_mp4, fourcc, fps, (w, h))
        if not writer.isOpened():
            log.error(
                "[%s] SAVE_ANNOTATED_VIDEO: could not open VideoWriter for %s "
                "(codec=%s fps=%.2f). Try SAVE_ANNOTATED_VIDEO_CODEC=mp4v",
                self.camera_id,
                out_mp4,
                self._annotated_video_codec,
                fps,
            )
            self._annotated_video_failed = True
            try:
                writer.release()
            except Exception:
                pass
            return
        self._annotated_video_writer = writer
        self._annotated_video_open_path = out_mp4
        log.info(
            "[%s] SAVE_ANNOTATED_VIDEO writing %s @ %.2f fps (%dx%d)",
            self.camera_id,
            out_mp4,
            fps,
            w,
            h,
        )

    def _annotated_video_write_frame(self, annotated_bgr: Any, fps_measured: float) -> None:
        if not self._save_annotated_video or self._annotated_video_failed:
            return
        self._ensure_annotated_video_writer(annotated_bgr, fps_measured)
        if self._annotated_video_writer is None:
            return
        try:
            self._annotated_video_writer.write(annotated_bgr)
        except Exception:
            log.exception("[%s] SAVE_ANNOTATED_VIDEO write failed", self.camera_id)
            self._annotated_video_failed = True

    def _close_annotated_video_writer(self) -> None:
        if self._annotated_video_writer is None:
            return
        path = self._annotated_video_open_path
        try:
            self._annotated_video_writer.release()
        except Exception:
            log.exception("[%s] SAVE_ANNOTATED_VIDEO release failed", self.camera_id)
        self._annotated_video_writer = None
        self._annotated_video_open_path = None
        if path:
            log.info("[%s] SAVE_ANNOTATED_VIDEO closed %s", self.camera_id, path)

    def _connect_redis_with_retry(self) -> Optional[object]:
        if not _REDIS_AVAILABLE:
            return None
        delay = 0.5
        for attempt in range(self._redis_connect_retries):
            try:
                client = _redis_lib.Redis.from_url(
                    self._redis_url, socket_connect_timeout=2, socket_timeout=2
                )
                client.ping()
                return client
            except Exception as exc:
                log.warning(
                    "[%s] FrameBus Redis connect attempt %d/%d failed: %s",
                    self.camera_id,
                    attempt + 1,
                    self._redis_connect_retries,
                    exc,
                )
                time.sleep(delay)
                delay = min(delay * 2, 8.0)
        return None

    def _try_reconnect_redis(self) -> None:
        if self._redis is not None:
            return
        client = self._connect_redis_with_retry()
        if client is not None:
            self._redis = client
            self._redis_breaker.record_success()
            log.info("[%s] FrameBus: Redis reconnected", self.camera_id)

    def _publish_live_jpeg(
        self,
        annotated,
        *,
        task_encode_buf=None,
        reuse_task_encode: bool = False,
    ) -> float:
        """Publish annotated frame to Redis. Returns wall seconds spent in publish path."""
        if self._redis is None:
            return 0.0
        if not self._redis_breaker.allow_request():
            log.warning(
                "[%s] live frame publish skipped: redis circuit %s",
                self.camera_id,
                self._redis_breaker.state_label(),
            )
            return 0.0
        t0 = time.perf_counter()
        seq = next_seq(self._frame_seq_counter, self._frame_seq_lock)

        to_encode = annotated
        if self._live_scale < 0.999:
            h0, w0 = to_encode.shape[:2]
            nw = max(2, int(w0 * self._live_scale))
            nh = max(2, int(h0 * self._live_scale))
            to_encode = cv2.resize(to_encode, (nw, nh), interpolation=cv2.INTER_AREA)

        reuse_ok = (
            reuse_task_encode
            and task_encode_buf is not None
            and self._task_jpeg_quality == self._live_jpeg_quality
            and self._live_scale >= 0.999
        )
        try:
            if reuse_ok:
                _buf = task_encode_buf
            else:
                _, _buf = cv2.imencode(
                    ".jpg",
                    to_encode,
                    [cv2.IMWRITE_JPEG_QUALITY, self._live_jpeg_quality],
                )
            jpeg_b64 = base64.b64encode(bytes(_buf)).decode("ascii")
            envelope = json.dumps({"_seq": seq, "jpeg": jpeg_b64}, separators=(",", ":"))
        except Exception:
            log.exception("[%s] live JPEG encode failed", self.camera_id)
            return time.perf_counter() - t0

        if not self._publish_frame_breaker.should_publish():
            self._publish_frame_breaker.buffer_envelope(envelope)
            return time.perf_counter() - t0

        def _pub_one(env: str) -> None:
            self._redis.publish(f"live:frame:{self.camera_id}", env)

        try:
            pending = self._publish_frame_breaker.drain_buffer()
            for env in pending:
                _pub_one(env)
            _pub_one(envelope)
            self._publish_frame_breaker.record_success()
            self._redis_breaker.record_success()
            if seq is not None:
                self._last_live_publish_seq = int(seq)
        except Exception:
            self._publish_frame_breaker.record_failure()
            self._publish_frame_breaker.buffer_envelope(envelope)
            self._redis_breaker.record_failure()
            log.exception("[%s] live frame publish failed", self.camera_id)
            try:
                self._redis.close()
            except Exception:
                pass
            self._redis = None
        return time.perf_counter() - t0

    def _env_float_clamped(self, name: str, default: float, minimum: float, maximum: float) -> float:
        try:
            value = float(os.getenv(name, str(default)))
        except ValueError:
            value = default
        return min(max(value, minimum), maximum)

    def _record_integrity(self, corrupt: bool) -> None:
        t = time.time()
        self._integrity_events.append((t, corrupt))
        while self._integrity_events and t - self._integrity_events[0][0] > 5.0:
            self._integrity_events.popleft()

    def _corrupt_rate_5s(self) -> float:
        if not self._integrity_events:
            return 0.0
        bad = sum(1 for _, c in self._integrity_events if c)
        return bad / len(self._integrity_events)

    def _ws_backpressure_flag(self) -> bool:
        if self._redis is None:
            return False
        try:
            v = self._redis.get(f"stream:ws_backpressure:{self.camera_id}")
            if v is None:
                return False
            if isinstance(v, bytes):
                return v not in (b"0", b"", b"false")
            return str(v).lower() not in ("0", "", "false")
        except Exception:
            return False

    def _publish_quality_tier_redis(self) -> None:
        row = LIVE_QUALITY_LADDER[self._live_quality_tier]
        self._live_jpeg_quality = int(row[0])
        self._live_scale = float(row[1])
        label = str(row[2])
        if self._redis is None:
            return
        try:
            self._redis.set(f"stream:quality:{self.camera_id}", label, ex=86400)
        except Exception:
            log.warning("[%s] stream:quality Redis SET failed", self.camera_id)

    def _maybe_adjust_live_quality_tier(self, corrupt_rate: float, ws_bp: bool) -> None:
        now = time.time()
        stressed = corrupt_rate > 0.2 or ws_bp
        relaxed = corrupt_rate < 0.05 and not ws_bp

        if stressed:
            if self._above_bad_since is None:
                self._above_bad_since = now
            self._below_good_since = None
        else:
            self._above_bad_since = None
            if relaxed:
                if self._below_good_since is None:
                    self._below_good_since = now
            else:
                self._below_good_since = None

        idx = self._live_quality_tier
        if (
            self._above_bad_since is not None
            and now - self._above_bad_since >= 3.0
            and idx < len(LIVE_QUALITY_LADDER) - 1
        ):
            self._live_quality_tier = idx + 1
            self._above_bad_since = now
            log.info(
                "[%s] Live quality stepped down to tier %s",
                self.camera_id,
                LIVE_QUALITY_LADDER[self._live_quality_tier][2],
            )
            self._publish_quality_tier_redis()
            return

        if (
            self._below_good_since is not None
            and now - self._below_good_since >= 10.0
            and idx > 0
        ):
            self._live_quality_tier = idx - 1
            self._below_good_since = now
            log.info(
                "[%s] Live quality stepped up to tier %s",
                self.camera_id,
                LIVE_QUALITY_LADDER[self._live_quality_tier][2],
            )
            self._publish_quality_tier_redis()

    def _coalesce_oldest_task_payload(self, q, tid: str, force: bool = False) -> bool:
        if not self._task_queue_coalesce:
            return False

        if not force:
            if self._task_queue_coalesce_threshold >= 1.0:
                return False
            try:
                queue_usage_ratio = q.qsize() / self._task_queue_maxsize
            except Exception:
                return False
            if queue_usage_ratio < self._task_queue_coalesce_threshold:
                return False

        try:
            q.get_nowait()
            self._task_queue_coalesced_by_task[tid] = (
                self._task_queue_coalesced_by_task.get(tid, 0) + 1
            )
            return True
        except _queue.Empty:
            return False
        except Exception:
            return False

    def _enqueue_task_payload(self, q, tid: str, payload: dict) -> None:
        """
        Bounded queue fan-out. When full, optionally drop the oldest item and
        retry once so workers stay on fresh frames (TASK_QUEUE_COALESCE).
        """
        self._coalesce_oldest_task_payload(q, tid)
        try:
            q.put_nowait(payload)
            return
        except _queue.Full:
            pass
        except Exception:
            self._frames_dropped += 1
            self._task_queue_drops_by_task[tid] = (
                self._task_queue_drops_by_task.get(tid, 0) + 1
            )
            self._warn_task_queue_full(tid)
            return

        self._coalesce_oldest_task_payload(q, tid, force=True)
        try:
            q.put_nowait(payload)
            return
        except _queue.Full:
            pass

        self._frames_dropped += 1
        self._task_queue_drops_by_task[tid] = (
            self._task_queue_drops_by_task.get(tid, 0) + 1
        )
        self._warn_task_queue_full(tid)

    def _warn_task_queue_full(self, tid: str) -> None:
        nowt = time.time()
        if nowt - self._queue_warn_last.get(tid, 0.0) >= self._queue_warn_interval:
            self._queue_warn_last[tid] = nowt
            hint = (
                "Consider raising TASK_QUEUE_MAXSIZE, lowering WIDTH, using a "
                "lower-resolution RTSP substream, or keep TASK_QUEUE_COALESCE=true "
                "(drops oldest queued frame for freshest)."
            )
            log.warning(
                "[%s] Task queue saturated; dropping frame for task %s "
                "(TASK_QUEUE_MAXSIZE=%s). %s",
                self.camera_id,
                tid,
                self._task_queue_maxsize,
                hint,
            )

    def _draw_boxes_opencv(self, frame, detections: list):
        """Lighter than Ultralytics plot(); for Redis preview / LIVE_ANNOTATION_MODE=opencv."""
        out = frame
        for det in detections:
            x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), 1, lineType=cv2.LINE_AA)
            tid = det.track_id
            if tid is not None and int(tid) >= 0:
                label = f"{det.class_name} {int(tid)}"
            else:
                label = str(det.class_name)
            cv2.putText(
                out,
                label,
                (x1, max(0, y1 - 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 255, 0),
                1,
                lineType=cv2.LINE_AA,
            )
        return out

    def _annotate_for_stream(
        self,
        resized_frame,
        results,
        detections: list,
        need_draw: bool,
        frame_count: int,
    ):
        """
        Return ``(annotated_frame, had_boxes)`` for every consumer
        (Redis live publish, disk save, scene evidence).

        Rule: **if detections > 0, always annotate — no exceptions**.

        * ``LIVE_ANNOTATION_MODE=ultralytics`` → Ultralytics ``plot()`` (rich).
        * ``LIVE_ANNOTATION_MODE=opencv`` or ``none``      → lightweight OpenCV overlay.
        * No detections → raw frame returned as-is (no draw cost, no stale buf risk).
        * ``need_draw=False`` (no Redis, no SAVE_OUTPUT) → skip draw only when
          there are genuinely no detections; if there *are* detections we still draw
          so any future publish/save path never receives a raw frame.
        """
        if frame_count % self._annotation_debug_every == 0:
            log.debug(
                "[%s] annotation: frame=%s detections=%d need_draw=%s mode=%s",
                self.camera_id,
                frame_count,
                len(detections),
                need_draw,
                self._live_annotation_mode,
            )

        if not detections:
            return resized_frame, False

        # Detections present — always annotate.
        if self._live_annotation_mode == "ultralytics" and results:
            return results[0].plot(), True
        return self._draw_boxes_opencv(resized_frame, detections), True

    def _draw_live_stream_geometry(self, bgr: np.ndarray) -> None:
        """Draw cross-line segments and cashier ROI outlines in-place on ``bgr``."""
        spec = self._live_overlay
        if not spec:
            return
        for ln in spec.get("cross_lines") or []:
            try:
                p0 = (int(ln["x0"]), int(ln["y0"]))
                p1 = (int(ln["x1"]), int(ln["y1"]))
            except (KeyError, TypeError, ValueError):
                continue
            cv2.line(bgr, p0, p1, (0, 255, 255), 2, cv2.LINE_AA)
            label = str(ln.get("label") or "")
            if label:
                mx = (p0[0] + p1[0]) // 2
                my = (p0[1] + p1[1]) // 2
                cv2.putText(
                    bgr,
                    label,
                    (mx, max(14, my - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
        h, w = bgr.shape[:2]
        if h < 2 or w < 2:
            return
        for zn in spec.get("cashier_zones") or []:
            pts = zn.get("points_norm") or []
            if len(pts) < 2:
                continue
            pix: List[Tuple[int, int]] = []
            ok = True
            for p in pts:
                if not isinstance(p, (list, tuple)) or len(p) < 2:
                    ok = False
                    break
                try:
                    fx = float(p[0])
                    fy = float(p[1])
                except (TypeError, ValueError):
                    ok = False
                    break
                pix.append((int(fx * w), int(fy * h)))
            if not ok or len(pix) < 2:
                continue
            arr = np.array(pix, dtype=np.int32).reshape((-1, 1, 2))
            color = zn.get("color_bgr") or (0, 200, 100)
            try:
                cb, cg, cr = int(color[0]), int(color[1]), int(color[2])
            except (TypeError, ValueError, IndexError):
                cb, cg, cr = 0, 200, 100
            cv2.polylines(bgr, [arr], True, (cb, cg, cr), 2, cv2.LINE_AA)
            zlabel = str(zn.get("label") or "")
            if zlabel:
                ax, ay = int(pix[0][0]), int(pix[0][1])
                cv2.putText(
                    bgr,
                    zlabel,
                    (ax + 4, min(h - 2, ay + 18)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (cb, cg, cr),
                    1,
                    cv2.LINE_AA,
                )

    def run(self):
        from stream import QUALITY_LADDER, StreamExhausted, StreamGeneratorMetrics, frames

        fps_counter      = 0
        fps_timer        = time.time()
        started_at       = time.time()
        frame_count      = 0
        total_detections = 0
        fps              = 0.0
        last_state_push_frame = 0
        last_state_push_wall = 0.0

        log.info(
            "[%s] FrameBus started — tasks: %s",
            self.camera_id,
            list(self.task_queues.keys()),
        )

        _q0 = QUALITY_LADDER[0]["label"] if QUALITY_LADDER else "live"
        self.shared_state[self.camera_id] = {
            "camera_id"       : self.camera_id,
            "rtsp_url"        : self.rtsp_url,
            "running"         : True,
            "frame_count"     : 0,
            "fps"             : 0.0,
            "last_detections" : 0,
            "total_detections": 0,
            "uptime_seconds"  : 0.0,
            "error"           : None,
            "stream_quality"  : _q0,
            "frames_dropped"  : 0,
            "drop_rate"       : 0.0,
            "decode_error_rate": 0.0,
            "task_queue_drops": 0,
            "task_queue_drop_rate": 0.0,
            "reconnects"      : 0,
            "latency_estimate_ms": 0.0,
            "stream_read_failures": 0,
            "decode_failures" : 0,
            "decoder"         : "cpu",
            "hw_decoder_requested": None,
            "hw_decoder_active": False,
            "stream_codec"    : None,
            "stream_native"   : None,
            "stream_target"   : None,
            "stream_health"   : {},
            "profile"         : os.getenv("RTSP_PROFILE", "balanced").lower(),
            "transport"       : os.getenv("RTSP_TRANSPORT", "tcp"),
            "embed_skip_rate" : 0.0,
            "uptime_sec"      : 0.0,
            "fps_actual"      : 0.0,
            "state_updated_at": time.time(),
            "task_queue_drops_by_task": dict(self._task_queue_drops_by_task),
            "task_queue_coalesced_by_task": dict(self._task_queue_coalesced_by_task),
            "live_annotation_mode": self._live_annotation_mode,
            "rtsp_backend": "unknown",
            "redis_circuit_state": self._redis_breaker.state_label(),
            "stream_metrics": {},
            "events_buffered": 0,
            "events_replayed": 0,
            "respawn_count": 0,
            "task_redis_circuit_state": None,
            "save_output": self.save_output,
            "save_annotated_video": self._save_annotated_video,
            "annotated_video_path": None,
            "redis_connected": self._redis is not None,
            "last_live_publish_seq": 0,
            "last_live_frame_had_boxes": False,
            "live_quality_tier": LIVE_QUALITY_LADDER[self._live_quality_tier][2],
            "live_scale": self._live_scale,
            "corrupt_rate_5s": 0.0,
            "live_stream_geometry_overlay": self._live_stream_geometry_active,
        }
        state_carry: Dict[str, Any] = dict(self.shared_state[self.camera_id])

        if self.save_output:
            os.makedirs(self.out_dir, exist_ok=True)

        corrupt_warn_last_wall = 0.0

        try:
            _stream_metrics = StreamGeneratorMetrics()
            for frame in frames(
                self.rtsp_url, camera_id=self.camera_id, metrics=_stream_metrics
            ):
                t_iter0 = time.perf_counter()
                now_wall = time.time()
                if self._metrics_last_wall_t is not None:
                    dt_ms = (now_wall - self._metrics_last_wall_t) * 1000.0
                    if self._metrics_inter_frame_ema_ms <= 0.0:
                        self._metrics_inter_frame_ema_ms = dt_ms
                    else:
                        self._metrics_inter_frame_ema_ms = (
                            0.85 * self._metrics_inter_frame_ema_ms + 0.15 * dt_ms
                        )
                self._metrics_last_wall_t = now_wall

                if self.stop_event.is_set():
                    break

                frame_count += 1
                fps_counter += 1
                self._frames_passed += 1

                elapsed = time.time() - fps_timer
                if elapsed >= 1.0:
                    fps         = round(fps_counter / elapsed, 2)
                    fps_counter = 0
                    fps_timer   = time.time()
                    # Recalculate publish cadence once we have a real FPS measurement
                    if fps > 0 and self._redis is not None:
                        self._publish_every = max(1, round(fps / self._target_fps))

                if frame_count % self._redis_health_interval == 0:
                    self._try_reconnect_redis()

                resized_frame = resize(frame, self.width, self.height)
                use_frame, was_corrupt = self._corruption.process(resized_frame)
                self._record_integrity(was_corrupt)
                corrupt_rate = self._corrupt_rate_5s()
                ws_bp = self._ws_backpressure_flag()
                self._maybe_adjust_live_quality_tier(corrupt_rate, ws_bp)

                if use_frame is None:
                    if now_wall - corrupt_warn_last_wall >= 5.0:
                        corrupt_warn_last_wall = now_wall
                        log.warning(
                            "[%s] Skipping frame (corruption heuristics, no prior good)",
                            self.camera_id,
                        )
                    continue
                resized_frame = use_frame

                if was_corrupt and now_wall - corrupt_warn_last_wall >= 5.0:
                    corrupt_warn_last_wall = now_wall
                    log.warning(
                        "[%s] Decode corruption heuristics: holding last good frame "
                        "(corrupt_rate_5s=%.2f)",
                        self.camera_id,
                        corrupt_rate,
                    )

                cc = self._corruption.consecutive_corrupt_all
                if cc > 10 and now_wall - self._last_idr_signal_wall >= 5.0:
                    self._last_idr_signal_wall = now_wall
                    try:
                        import stream as _sm

                        _sm.signal_adaptive_corruption_action(self.camera_id, idr=True)
                    except Exception:
                        log.debug("[%s] IDR signal skipped (stream module)", self.camera_id)

                if (
                    corrupt_rate > 0.3
                    and len(self._integrity_events) >= 20
                    and now_wall - self._last_keyframe_only_signal_wall >= 60.0
                ):
                    self._last_keyframe_only_signal_wall = now_wall
                    try:
                        import stream as _sm2

                        _sm2.signal_adaptive_corruption_action(
                            self.camera_id, keyframe_only=True
                        )
                    except Exception:
                        log.debug("[%s] keyframe-only signal skipped", self.camera_id)
                    log.warning(
                        "[%s] High corruption (%.0f%% over ~5s) — keyframe-only FFmpeg requested",
                        self.camera_id,
                        corrupt_rate * 100.0,
                    )

                t_after_resize = time.perf_counter()

                _, buf = cv2.imencode(
                    ".jpg",
                    resized_frame,
                    [cv2.IMWRITE_JPEG_QUALITY, self._task_jpeg_quality],
                )
                frame_b64 = base64.b64encode(buf).decode("utf-8")
                t_after_task_enc = time.perf_counter()

                # ── BoT-SORT tracking ──────────────────────────────────────────
                try:
                    _track_kw: Dict[str, Any] = {
                        "persist": True,  # keeps track state across frames
                        "tracker": self._tracker_yaml,
                        "conf": self._conf,
                        "iou": self._iou,
                        "classes": self._classes,
                        "device": self._device,
                        "verbose": False,
                        "max_det": self._max_det,
                    }
                    if self._imgsz is not None:
                        _track_kw["imgsz"] = self._imgsz
                    results = self._model.track(resized_frame, **_track_kw)
                except RuntimeError as _track_err:
                    log.error(
                        "[%s] model.track() error — resetting tracker state: %s",
                        self.camera_id, _track_err,
                    )
                    try:
                        self._model.reset()
                    except Exception:
                        pass
                    continue

                t_after_yolo = time.perf_counter()
                detections = self._parse_tracks(results)
                last_det   = len(detections)
                total_detections += last_det

                # ── Save best crop per tracked person ─────────────────────────
                self._save_best_crops(resized_frame, detections, frame_count)

                will_publish = self._redis is not None and (
                    frame_count % self._publish_every == 0
                )
                need_draw = self.save_output or will_publish or self._save_annotated_video

                # Single annotated frame used everywhere: Redis, disk, scene evidence.
                # detections > 0  → always returns an annotated copy (rule enforced in helper).
                # detections == 0 → returns raw frame; JPEG reuse is safe.
                annotated, had_boxes = self._annotate_for_stream(
                    resized_frame,
                    results,
                    detections,
                    need_draw,
                    frame_count,
                )
                if self._live_stream_geometry_active and need_draw:
                    self._draw_live_stream_geometry(annotated)
                t_after_ann = time.perf_counter()

                # Reuse the pre-track JPEG only when the annotated frame IS the raw
                # frame (no detections, no overlay drawn).
                reuse_live_buf = will_publish and not had_boxes

                publish_sec = 0.0
                if will_publish:
                    publish_sec = self._publish_live_jpeg(
                        annotated,
                        task_encode_buf=buf,
                        reuse_task_encode=reuse_live_buf,
                    )
                    self._last_live_frame_had_boxes = had_boxes

                payload = {
                    "camera_id" : self.camera_id,
                    "frame_id"  : frame_count,
                    "timestamp" : datetime.utcnow().isoformat(),
                    "frame_b64" : frame_b64,
                    "detection" : {
                        "items": detections,
                        "count": last_det,
                    },
                }
                if self._include_frame_ndarray:
                    payload["frame"] = resized_frame.copy()

                for task_id, q in self.task_queues.items():
                    tid = str(task_id)
                    self._enqueue_task_payload(q, tid, payload)

                if self.save_output:
                    save_frame(annotated, self.out_dir, frame_count)

                if self._save_annotated_video:
                    if self._annotated_video_match_ws:
                        if self._redis is not None and will_publish:
                            self._annotated_video_write_frame(annotated, fps)
                        elif self._redis is None:
                            self._annotated_video_write_frame(annotated, fps)
                    else:
                        self._annotated_video_write_frame(annotated, fps)

                t_iter_end = time.perf_counter()
                if frame_count % self._perf_log_every == 0:
                    log.info(
                        "[%s] perf frame=%s total_ms=%.1f yolo_ms=%.1f "
                        "task_encode_ms=%.1f annotate_ms=%.1f publish_ms=%.1f "
                        "detections=%d mode=%s reuse_live_buf=%s",
                        self.camera_id,
                        frame_count,
                        (t_iter_end - t_iter0) * 1000.0,
                        (t_after_yolo - t_after_task_enc) * 1000.0,
                        (t_after_task_enc - t_after_resize) * 1000.0,
                        (t_after_ann - t_after_yolo) * 1000.0,
                        publish_sec * 1000.0,
                        last_det,
                        self._live_annotation_mode,
                        reuse_live_buf,
                    )

                embed_total = self._embed_passed + self._embed_skipped
                skip_rate = (
                    round(self._embed_skipped / embed_total * 100, 1) if embed_total else 0.0
                )
                try:
                    import stream as _stream_mod

                    quality_label = getattr(
                        _stream_mod, "_current_quality_label", _q0
                    )
                    decode_failures = int(
                        getattr(_stream_mod, "_current_decode_failures", 0)
                    )
                    decoder = str(getattr(_stream_mod, "_current_decoder_type", "cpu"))
                    hw_decoder_requested = getattr(
                        _stream_mod, "_current_hw_decoder_requested", None
                    )
                    hw_decoder_active = bool(
                        getattr(_stream_mod, "_current_hw_decoder_active", False)
                    )
                    stream_health = dict(
                        getattr(_stream_mod, "_current_stream_health", {}) or {}
                    )
                    profile = str(getattr(_stream_mod, "_current_profile", "balanced"))
                    transport = str(getattr(_stream_mod, "_current_transport", "tcp"))
                    reconnects = int(getattr(_stream_mod, "_rtsp_reconnect_count", 0))
                    rtsp_backend = str(
                        getattr(_stream_mod, "_current_rtsp_backend", "unknown")
                    )
                except Exception:
                    quality_label = _q0
                    decode_failures = 0
                    decoder = "cpu"
                    hw_decoder_requested = None
                    hw_decoder_active = False
                    stream_health = {}
                    profile = os.getenv("RTSP_PROFILE", "balanced").lower()
                    transport = os.getenv("RTSP_TRANSPORT", "tcp")
                    reconnects = 0
                    rtsp_backend = "unknown"

                # Stream source: failed VideoCapture.read() vs successful yields (same process / one generator).
                stream_reads = decode_failures + frame_count
                decode_error_rate = (
                    round(decode_failures / stream_reads, 4) if stream_reads > 0 else 0.0
                )
                # Task workers saturated (queue full) — not the same as RTSP read loss.
                routed_frames = self._frames_passed + self._frames_dropped
                task_queue_drop_rate = (
                    round(self._frames_dropped / routed_frames, 4) if routed_frames > 0 else 0.0
                )
                uptime_sec = round(time.time() - started_at, 1)

                state_carry.update(
                    {
                        "frame_count"     : frame_count,
                        "fps"             : fps,
                        "fps_actual"      : fps,
                        "last_detections" : last_det,
                        "total_detections": total_detections,
                        "uptime_seconds"  : uptime_sec,
                        "uptime_sec"      : uptime_sec,
                        "stream_quality"  : quality_label,
                        "frames_dropped"  : self._frames_dropped,
                        "drop_rate"       : decode_error_rate,
                        "decode_error_rate": decode_error_rate,
                        "task_queue_drops": self._frames_dropped,
                        "task_queue_drop_rate": task_queue_drop_rate,
                        "task_queue_drops_by_task": dict(
                            self._task_queue_drops_by_task
                        ),
                        "task_queue_coalesced_by_task": dict(
                            self._task_queue_coalesced_by_task
                        ),
                        "reconnects"      : reconnects,
                        "latency_estimate_ms": round(
                            self._metrics_inter_frame_ema_ms, 2
                        ),
                        "stream_read_failures": decode_failures,
                        "decode_failures" : decode_failures,
                        "decoder"         : decoder,
                        "hw_decoder_requested": hw_decoder_requested,
                        "hw_decoder_active": hw_decoder_active,
                        "stream_codec"    : stream_health.get("codec"),
                        "stream_native"   : stream_health.get("native"),
                        "stream_target"   : stream_health.get("target"),
                        "stream_health"   : stream_health,
                        "profile"         : profile,
                        "transport"       : transport,
                        "rtsp_backend"    : rtsp_backend,
                        "embed_skip_rate" : skip_rate,
                        "state_updated_at": time.time(),
                        "live_annotation_mode": self._live_annotation_mode,
                        "redis_circuit_state": self._redis_breaker.state_label(),
                        "stream_metrics": {
                            "decode_failures": _stream_metrics.decode_failures,
                            "reconnect_count": _stream_metrics.reconnect_count,
                        },
                        "save_output": self.save_output,
                        "save_annotated_video": self._save_annotated_video,
                        "annotated_video_path": self._annotated_video_open_path,
                        "redis_connected": self._redis is not None,
                        "last_live_publish_seq": self._last_live_publish_seq,
                        "last_live_frame_had_boxes": self._last_live_frame_had_boxes,
                        "task_queue_jpeg_quality": self._task_jpeg_quality,
                        "live_jpeg_quality": self._live_jpeg_quality,
                        "live_quality_tier": LIVE_QUALITY_LADDER[self._live_quality_tier][2],
                        "live_scale": self._live_scale,
                        "corrupt_rate_5s": round(corrupt_rate, 4),
                        "live_stream_geometry_overlay": self._live_stream_geometry_active,
                    }
                )
                push_state = (
                    frame_count - last_state_push_frame >= self._state_update_every_n
                    or now_wall - last_state_push_wall >= self._state_update_min_sec
                )
                if push_state:
                    self.shared_state[self.camera_id] = state_carry
                    last_state_push_frame = frame_count
                    last_state_push_wall = now_wall

        except StreamExhausted as e:
            if self._bus_fatal_event is not None:
                self._bus_fatal_event.set()
            state_carry.update(
                {
                    "error": str(e),
                    "running": False,
                    "state_updated_at": time.time(),
                    "stopped_reason": "stream_exhausted",
                }
            )
            self.shared_state[self.camera_id] = state_carry
            log.warning("[%s] FrameBus stream exhausted: %s", self.camera_id, e)
        except Exception as e:
            if self._bus_fatal_event is not None:
                self._bus_fatal_event.set()
            state_carry.update(
                {
                    "error"  : str(e),
                    "running": False,
                    "state_updated_at": time.time(),
                }
            )
            self.shared_state[self.camera_id] = state_carry
            log.exception("[%s] FrameBus error: %s", self.camera_id, e)
        finally:
            try:
                if self._annotated_video_open_path:
                    state_carry["annotated_video_path"] = self._annotated_video_open_path
            except Exception:
                pass
            self._close_annotated_video_writer()
            try:
                state_carry.update(
                    {
                        "running": False,
                        "state_updated_at": time.time(),
                    }
                )
                self.shared_state[self.camera_id] = state_carry
            except Exception:
                self.shared_state[self.camera_id] = {
                    "camera_id": self.camera_id,
                    "running": False,
                    "state_updated_at": time.time(),
                }
            log.info(
                "[%s] FrameBus stopped. Frames: %s", self.camera_id, frame_count
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Best-crop-per-track: progressive overwrite
    # ─────────────────────────────────────────────────────────────────────────

    def _save_best_crops(self, frame, detections, frame_id: int):
        """
        For each tracked person, save the crop when bbox area exceeds previous best
        by 20%, and only if confidence and crop variance pass thresholds.
        Emits a message to the embedding_queue for EmbeddingWorker.
        """
        persons = [d for d in detections if d.class_name == "person" and d.track_id not in (None, -1)]
        current_track_ids = set()

        for det in persons:
            tid = det.track_id
            current_track_ids.add(tid)

            x1, y1, x2, y2 = det.bbox
            current_area = (x2 - x1) * (y2 - y1)

            state = self._track_state.get(tid, {"best_area": 0, "last_frame": frame_id})
            previous_best = state["best_area"]
            state["last_frame"] = frame_id

            # Only save when crop is 20% larger (or first time)
            if current_area > (previous_best * 1.2):
                crop = self._crop_bbox(frame, x1, y1, x2, y2)
                if crop.size == 0:
                    self._track_state[tid] = state
                    continue

                if det.confidence < _EMBED_CONF_THRESHOLD:
                    self._embed_skipped += 1
                    self._track_state[tid] = state
                    continue

                small = cv2.resize(crop, (80, 80), interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                if float(gray.var()) < _EMBED_MIN_VARIANCE:
                    self._embed_skipped += 1
                    self._track_state[tid] = state
                    continue

                # Laplacian sharpness gate — rejects motion-blurred crops before
                # they reach the ONNX embedding worker and corrupt the ReID gallery.
                if _EMBED_MIN_SHARPNESS > 0:
                    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                    if lap_var < _EMBED_MIN_SHARPNESS:
                        self._embed_skipped += 1
                        self._track_state[tid] = state
                        continue

                # Deterministic filename: 1 file per track per camera
                crop_path = os.path.join(self._crop_dir, f"track_{tid}.jpg")
                cv2.imwrite(crop_path, crop)

                if self.embedding_queue is not None:
                    msg = {
                        "camera_id" : self.camera_id,
                        "track_id"  : tid,
                        "crop_path" : crop_path,
                        "frame_id"  : frame_id,
                        "bbox"      : [x1, y1, x2, y2],
                        "confidence": det.confidence,
                        "timestamp" : datetime.utcnow().isoformat(),
                    }
                    try:
                        self.embedding_queue.put_nowait(msg)
                        self._embed_passed += 1
                    except Exception:
                        self._embed_skipped += 1

                state["best_area"] = current_area

            self._track_state[tid] = state

        # ── Cleanup stale tracks (not seen in 60 frames) ──
        stale = [
            t for t, s in self._track_state.items()
            if (frame_id - s["last_frame"] > 60) and (t not in current_track_ids)
        ]
        for t in stale:
            del self._track_state[t]

    def _crop_bbox(self, frame, x1: int, y1: int, x2: int, y2: int):
        """Crop bounding box with padding, clamped to frame bounds."""
        h, w = frame.shape[:2]
        x1 = max(0, x1 - self._padding)
        y1 = max(0, y1 - self._padding)
        x2 = min(w, x2 + self._padding)
        y2 = min(h, y2 + self._padding)
        return frame[y1:y2, x1:x2]

    # ─────────────────────────────────────────────────────────────────────────

    def _parse_tracks(self, results) -> list:
        """Convert YOLO track results into Detection objects with track_id set."""
        if not results or results[0].boxes is None:
            return []

        boxes     = results[0].boxes
        has_ids   = boxes.id is not None

        detections = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(int, boxes.xyxy[i])
            cls_id   = int(boxes.cls[i])
            conf     = float(boxes.conf[i])
            track_id = int(boxes.id[i]) if has_ids else -1

            detections.append(Detection(
                x1         = x1,
                y1         = y1,
                x2         = x2,
                y2         = y2,
                class_id   = cls_id,
                class_name = self._names[cls_id],
                confidence = conf,
                track_id   = track_id,
            ))

        return detections
