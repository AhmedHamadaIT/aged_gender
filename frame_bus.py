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

import time
import base64
import hashlib
from datetime import datetime
from typing import Dict, Optional

import cv2
from ultralytics import YOLO

from utils import resize, save_frame
from services.detector import Detection

try:
    import redis as _redis_lib
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False

# Minimum confidence / variance to emit crop to embedding worker
_EMBED_CONF_THRESHOLD = float(os.getenv("EMBED_CONF_THRESHOLD", "0.45"))
_EMBED_MIN_VARIANCE = float(os.getenv("FRAME_MIN_VARIANCE", "8.0"))


class FrameBus:
    def __init__(
        self,
        camera_id       : str,
        rtsp_url        : str,
        shared_state,
        stop_event,
        task_queues     : Dict[str, object],  # {task_id: Queue}
        embedding_queue = None,                # Queue for EmbeddingWorker
    ):
        self.camera_id       = camera_id
        self.rtsp_url        = rtsp_url
        self.shared_state    = shared_state
        self.stop_event      = stop_event
        self.task_queues     = task_queues
        self.embedding_queue = embedding_queue

        self.save_output = os.getenv("SAVE_OUTPUT", "True").lower() in ("true", "1", "yes")
        self.out_dir     = os.path.join(os.getenv("OUTPUT_DIR", "./outputs"), camera_id)
        self.width       = int(os.getenv("WIDTH",  "1280"))
        self.height      = int(os.getenv("HEIGHT", "0"))
        self._padding    = int(os.getenv("REID_PADDING", "10"))

        model_path   = os.getenv("YOLO_MODEL", "yolov8n.pt")
        conf         = float(os.getenv("CONF_THRESHOLD", "0.35"))
        _device_raw  = os.getenv("DEVICE", "0")
        device       = int(_device_raw) if _device_raw.isdigit() else _device_raw
        _classes_raw = os.getenv("FILTER_CLASSES", "")
        classes      = [int(c.strip()) for c in _classes_raw.split(",") if c.strip()] or None

        self._model   = YOLO(model_path, task="detect")
        self._conf    = conf
        self._device  = device
        self._classes = classes
        self._names   = self._model.names

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
        if _REDIS_AVAILABLE:
            try:
                redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
                self._redis = _redis_lib.Redis.from_url(redis_url, socket_connect_timeout=2)
                self._redis.ping()
                print(f"[{camera_id}] FrameBus: Redis connected ({redis_url})")
            except Exception as exc:
                print(f"[{camera_id}] FrameBus: Redis unavailable — live stream disabled ({exc})")
                self._redis = None

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

    def run(self):
        from stream import QUALITY_LADDER, frames

        fps_counter      = 0
        fps_timer        = time.time()
        started_at       = time.time()
        frame_count      = 0
        total_detections = 0
        fps              = 0.0

        print(f"[{self.camera_id}] FrameBus started — tasks: {list(self.task_queues.keys())}")

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
            "decode_failures" : 0,
            "decoder"         : "cpu",
            "hw_decoder_requested": None,
            "hw_decoder_active": False,
            "profile"         : os.getenv("RTSP_PROFILE", "balanced").lower(),
            "transport"       : os.getenv("RTSP_TRANSPORT", "tcp"),
            "embed_skip_rate" : 0.0,
        }

        if self.save_output:
            os.makedirs(self.out_dir, exist_ok=True)

        try:
            for frame in frames(self.rtsp_url, camera_id=self.camera_id):
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

                resized_frame = resize(frame, self.width, self.height)

                _, buf    = cv2.imencode(".jpg", resized_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_b64 = base64.b64encode(buf).decode("utf-8")

                # ── BoT-SORT tracking ──────────────────────────────────────────
                results = self._model.track(
                    resized_frame,
                    persist  = True,          # keeps track state across frames
                    tracker  = "botsort.yaml",
                    conf     = self._conf,
                    classes  = self._classes,
                    device   = self._device,
                    verbose  = False,
                )

                detections = self._parse_tracks(results)
                last_det   = len(detections)
                total_detections += last_det

                # ── Save best crop per tracked person ─────────────────────────
                self._save_best_crops(resized_frame, detections, frame_count)

                # Always annotate — needed for live stream even when SAVE_OUTPUT is off
                annotated = results[0].plot() if results else resized_frame

                # ── Publish annotated JPEG to Redis (live stream) ──────────────
                if self._redis is not None and frame_count % self._publish_every == 0:
                    try:
                        _, _buf = cv2.imencode(
                            ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75]
                        )
                        self._redis.publish(
                            f"live:frame:{self.camera_id}", bytes(_buf)
                        )
                    except Exception:
                        pass  # never block inference on Redis errors

                payload = {
                    "camera_id" : self.camera_id,
                    "frame_id"  : frame_count,
                    "timestamp" : datetime.utcnow().isoformat(),
                    "frame_b64" : frame_b64,
                    "frame"     : resized_frame.copy(),
                    "detection" : {
                        "items": detections,
                        "count": last_det,
                    },
                }

                for q in self.task_queues.values():
                    try:
                        q.put_nowait(payload)
                    except Exception:
                        self._frames_dropped += 1

                if self.save_output:
                    save_frame(annotated, self.out_dir, frame_count)

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
                    profile = str(getattr(_stream_mod, "_current_profile", "balanced"))
                    transport = str(getattr(_stream_mod, "_current_transport", "tcp"))
                except Exception:
                    quality_label = _q0
                    decode_failures = 0
                    decoder = "cpu"
                    hw_decoder_requested = None
                    hw_decoder_active = False
                    profile = os.getenv("RTSP_PROFILE", "balanced").lower()
                    transport = os.getenv("RTSP_TRANSPORT", "tcp")

                routed_frames = self._frames_passed + self._frames_dropped
                drop_rate = (
                    round(self._frames_dropped / routed_frames * 100, 2)
                    if routed_frames
                    else 0.0
                )

                self.shared_state[self.camera_id] = {
                    **self.shared_state[self.camera_id],
                    "frame_count"     : frame_count,
                    "fps"             : fps,
                    "last_detections" : last_det,
                    "total_detections": total_detections,
                    "uptime_seconds"  : round(time.time() - started_at, 1),
                    "stream_quality"  : quality_label,
                    "frames_dropped"  : self._frames_dropped,
                    "drop_rate"       : drop_rate,
                    "decode_failures" : decode_failures,
                    "decoder"         : decoder,
                    "hw_decoder_requested": hw_decoder_requested,
                    "hw_decoder_active": hw_decoder_active,
                    "profile"         : profile,
                    "transport"       : transport,
                    "embed_skip_rate" : skip_rate,
                }

        except Exception as e:
            self.shared_state[self.camera_id] = {
                **self.shared_state[self.camera_id],
                "error"  : str(e),
                "running": False,
            }
            print(f"[{self.camera_id}] FrameBus error: {e}")
        finally:
            self.shared_state[self.camera_id] = {
                **self.shared_state[self.camera_id],
                "running": False,
            }
            print(f"[{self.camera_id}] FrameBus stopped. Frames: {frame_count}")

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
