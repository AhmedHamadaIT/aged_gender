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
import time
import base64
import hashlib
from datetime import datetime
from typing import Dict, Optional

import cv2
from ultralytics import YOLO

from utils import resize, save_frame
from services.detector import Detection


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

    def run(self):
        from stream import frames

        fps_counter      = 0
        fps_timer        = time.time()
        started_at       = time.time()
        frame_count      = 0
        total_detections = 0
        fps              = 0.0

        print(f"[{self.camera_id}] FrameBus started — tasks: {list(self.task_queues.keys())}")

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
        }

        if self.save_output:
            os.makedirs(self.out_dir, exist_ok=True)

        try:
            for frame in frames(self.rtsp_url):
                if self.stop_event.is_set():
                    break

                frame_count += 1
                fps_counter += 1

                elapsed = time.time() - fps_timer
                if elapsed >= 1.0:
                    fps         = round(fps_counter / elapsed, 2)
                    fps_counter = 0
                    fps_timer   = time.time()

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

                annotated = results[0].plot() if self.save_output and results else resized_frame

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
                        pass  # drop frame if task is backlogged — never block capture

                if self.save_output:
                    save_frame(annotated, self.out_dir, frame_count)

                self.shared_state[self.camera_id] = {
                    **self.shared_state[self.camera_id],
                    "frame_count"     : frame_count,
                    "fps"             : fps,
                    "last_detections" : last_det,
                    "total_detections": total_detections,
                    "uptime_seconds"  : round(time.time() - started_at, 1),
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
        For each tracked person, save the crop only when the bounding box area
        exceeds the previous best by 20%. Emits a message to the embedding_queue
        so the EmbeddingWorker can extract embeddings asynchronously.
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

                # Deterministic filename: 1 file per track per camera
                crop_path = os.path.join(self._crop_dir, f"track_{tid}.jpg")
                cv2.imwrite(crop_path, crop)

                # Emit to embedding worker (non-blocking)
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
                    except Exception:
                        pass  # drop if worker is backlogged — crop is on disk

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
