"""
pipeline.py
-----------
CameraPipeline — runs inside each camera process.

Context structure passed between services:
{
    "data": {
        "frame"    : np.ndarray,          ← raw frame
        "detection": { "items": [...], "count": N },   ← set by DetectorService
        "use_case" : {}                   ← set by downstream use-case services
    }
}
"""

import os
import time
from typing import List, Type

from utils import resize, annotate, save_frame

from logger.logger_config import Logger
import os
from dotenv import load_dotenv
load_dotenv()
log = Logger.get_logger(__name__)


class CameraPipeline:
    """
    Runs all registered services on each frame for a single camera.

    Usage:
        pipeline = CameraPipeline(camera_id, rtsp_url, shared_state, stop_event)
        pipeline.register(DetectorService)
        pipeline.register(CountingService)
        pipeline.run()
    """

    def __init__(self, camera_id: str, rtsp_url: str, shared_state, stop_event):
        self.camera_id    = camera_id
        self.rtsp_url     = rtsp_url
        self.shared_state = shared_state
        self.stop_event   = stop_event
        self._service_classes: List[Type] = []

        self.save_output = os.getenv("SAVE_OUTPUT", "True").lower() in ("true", "1", "yes")
        self.out_dir     = os.path.join(os.getenv("OUTPUT_DIR", "./outputs"), camera_id)
        self.width       = int(os.getenv("WIDTH",  "1280"))
        self.height      = int(os.getenv("HEIGHT", "0"))

    def register(self, service_class: Type):
        """Register a service class to run on each frame."""
        self._service_classes.append(service_class)

    def run(self):
        """Start the pipeline loop. Blocks until stop_event is set."""
        from stream import frames

        services   = [cls() for cls in self._service_classes]
        fps_counter = 0
        fps_timer   = time.time()
        started_at  = time.time()
        frame_count = 0
        fps         = 0.0

        log.info(f"[{self.camera_id}] Pipeline started — "
              f"services: {[cls.__name__ for cls in self._service_classes]}")

        self.shared_state[self.camera_id] = {
            "camera_id"    : self.camera_id,
            "rtsp_url"     : self.rtsp_url,
            "running"      : True,
            "frame_count"  : 0,
            "fps"          : 0.0,
            "uptime_seconds": 0.0,
            "error"        : None,
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

                # ── Build initial context ──
                context = {
                    "data": {
                        "frame"    : resize(frame, self.width, self.height),
                        "detection": {},
                        "use_case" : {},
                    }
                }

                # ── Run all services ──
                for service in services:
                    context = service(context)

                # ── Save annotated frame ──
                if self.save_output:
                    detections = context["data"].get("detection", {}).get("items", [])
                    annotated  = annotate(context["data"]["frame"], detections, fps, self.camera_id)
                    save_frame(annotated, self.out_dir, frame_count)

                # ── Update shared state ──
                self.shared_state[self.camera_id] = {
                    **self.shared_state[self.camera_id],
                    "frame_count"  : frame_count,
                    "fps"          : fps,
                    "uptime_seconds": round(time.time() - started_at, 1),
                }

        except Exception as e:
            self.shared_state[self.camera_id] = {
                **self.shared_state[self.camera_id],
                "error"  : str(e),
                "running": False,
            }
            log.info(f"[{self.camera_id}] Pipeline error: {e}")
        finally:
            self.shared_state[self.camera_id] = {
                **self.shared_state[self.camera_id],
                "running": False,
            }
            log.info(f"[{self.camera_id}] Pipeline stopped. Frames: {frame_count}")