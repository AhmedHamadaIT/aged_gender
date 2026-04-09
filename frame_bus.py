"""
frame_bus.py
------------
FrameBus — runs inside each camera process.

Captures frames, runs YOLO BoT-SORT tracking, then fans out
{frame + tracked detections} to each registered task queue.

Track IDs are assigned here and carried on each Detection object,
so task workers never need to run their own detector or tracker.
"""

import os
import time
import base64
from datetime import datetime
from typing import Dict

import cv2
from ultralytics import YOLO

from utils import resize, save_frame
from services.detector import Detection


class FrameBus:
    def __init__(
        self,
        camera_id   : str,
        rtsp_url    : str,
        shared_state,
        stop_event,
        task_queues : Dict[str, object],  # {task_id: Queue}
    ):
        self.camera_id    = camera_id
        self.rtsp_url     = rtsp_url
        self.shared_state = shared_state
        self.stop_event   = stop_event
        self.task_queues  = task_queues

        self.save_output = os.getenv("SAVE_OUTPUT", "True").lower() in ("true", "1", "yes")
        self.out_dir     = os.path.join(os.getenv("OUTPUT_DIR", "./outputs"), camera_id)
        self.width       = int(os.getenv("WIDTH",  "1280"))
        self.height      = int(os.getenv("HEIGHT", "0"))

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
