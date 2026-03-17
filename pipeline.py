"""
pipeline.py
-----------
CameraPipeline — runs inside each camera process.

Context structure passed between services:
{
    "data": {
        "frame"    : np.ndarray,
        "detection": {"items": List[Detection], "count": int},
        "use_case" : {}
    }
}

Each frame result is pushed to the shared result_queue as:
{
    "camera_id"  : str,
    "frame_count": int,
    "timestamp"  : str,
    "frame"      : str,   ← base64 encoded raw JPEG (before annotation)
    "data": {
        "detection": {"items": [...], "count": N},
        "use_case" : {...}
    }
}
"""

import os
import time
import base64
import json
from datetime import datetime
from typing import List, Type

import cv2
import numpy as np

from utils import resize, save_frame


class CameraPipeline:
    def __init__(
        self,
        camera_id   : str,
        rtsp_url    : str,
        shared_state,
        stop_event,
        result_queue,
        pipeline_names: List[str],
    ):
        self.camera_id      = camera_id
        self.rtsp_url       = rtsp_url
        self.shared_state   = shared_state
        self.stop_event     = stop_event
        self.result_queue   = result_queue
        self.pipeline_names = pipeline_names
        self._service_classes: List[Type] = []

        self.save_output = os.getenv("SAVE_OUTPUT", "True").lower() in ("true", "1", "yes")
        self.out_dir     = os.path.join(os.getenv("OUTPUT_DIR", "./outputs"), camera_id)
        self.width       = int(os.getenv("WIDTH",  "1280"))
        self.height      = int(os.getenv("HEIGHT", "0"))

    def register(self, service_class: Type):
        self._service_classes.append(service_class)

    def run(self):
        from stream import frames

        services    = [cls() for cls in self._service_classes]
        fps_counter = 0
        fps_timer   = time.time()
        started_at  = time.time()
        frame_count = 0
        fps         = 0.0

        print(
            f"[{self.camera_id}] Pipeline started — "
            f"services: {[cls.__name__ for cls in self._service_classes]}"
        )

        total_detections = 0
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

                # ── Encode raw frame as base64 BEFORE annotation ──
                _, buf      = cv2.imencode(".jpg", resized_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_b64   = base64.b64encode(buf).decode("utf-8")

                # ── Build context ──
                context = {
                    "data": {
                        "frame"    : resized_frame.copy(),
                        "detection": {},
                        "use_case" : {},
                    }
                }

                # ── Run all services ──
                for service in services:
                    context = service(context)

                # ── Save annotated frame ──
                if self.save_output:
                    save_frame(context["data"]["frame"], self.out_dir, frame_count)

                # ── Build result payload ──
                detection_data = context["data"].get("detection", {})
                use_case_data  = context["data"].get("use_case",  {})

                result = {
                    "camera_id"  : self.camera_id,
                    "frame_count": frame_count,
                    "timestamp"  : datetime.utcnow().isoformat(),
                    "frame"      : frame_b64,
                    "data": {
                        "detection": {
                            "count": detection_data.get("count", 0),
                            "items": [
                                d.to_dict()
                                for d in detection_data.get("items", [])
                            ],
                        },
                        "use_case": {
                            k: [r.to_dict() for r in v] if isinstance(v, list) else v
                            for k, v in use_case_data.items()
                        },
                    },
                }

                # ── Push to result queue ──
                try:
                    self.result_queue.put_nowait(result)
                except Exception:
                    pass  # drop frame if queue is full — never block inference

                count = detection_data.get("count", 0)
                total_detections += count
                # ── Update shared state ──
                self.shared_state[self.camera_id] = {
                    **self.shared_state[self.camera_id],
                    "frame_count"   : frame_count,
                    "fps"           : fps,
                    "last_detections" : count,
                    "total_detections": total_detections,
                    "uptime_seconds": round(time.time() - started_at, 1),
                }

        except Exception as e:
            self.shared_state[self.camera_id] = {
                **self.shared_state[self.camera_id],
                "error"  : str(e),
                "running": False,
            }
            print(f"[{self.camera_id}] Pipeline error: {e}")
        finally:
            self.shared_state[self.camera_id] = {
                **self.shared_state[self.camera_id],
                "running": False,
            }
            print(f"[{self.camera_id}] Pipeline stopped. Frames: {frame_count}")