"""
pipeline.py
-----------
CameraPipeline — runs inside each camera process.

Each frame flows through all registered services in order.
Results from each service are passed as context to the next.

Pipeline context per frame:
{
    "frame"      : np.ndarray,
    "detections" : List[Detection],   ← set by DetectorService
    "counts"     : dict,              ← set by CountingService (future)
    "tracks"     : dict,              ← set by TrackingService (future)
    ...
}
"""

import os
import time
from typing import List, Type

import numpy as np

from utils import resize, annotate, save_frame


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

        # Instantiate all services fresh inside this process
        services = [cls() for cls in self._service_classes]

        print(f"[{self.camera_id}] Pipeline started with services: "
              f"{[cls.__name__ for cls in self._service_classes]}")

        fps_counter = 0
        fps_timer   = time.time()
        started_at  = time.time()
        frame_count = 0
        fps         = 0.0

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

                # ── Resize ──
                infer_frame = resize(frame, self.width, self.height)

                # ── Run pipeline ──
                context = {"frame": infer_frame}
                for service in services:
                    context = service(context)

                # ── Extract detection count for status ──
                detections = context.get("detections", [])

                # ── Save annotated frame ──
                if self.save_output:
                    annotated = annotate(infer_frame, detections, fps, self.camera_id)
                    save_frame(annotated, self.out_dir, frame_count)

                # ── Update shared state ──
                state = self.shared_state[self.camera_id]
                self.shared_state[self.camera_id] = {
                    **state,
                    "frame_count"     : frame_count,
                    "fps"             : fps,
                    "last_detections" : len(detections),
                    "total_detections": state["total_detections"] + len(detections),
                    "uptime_seconds"  : round(time.time() - started_at, 1),
                }

        except Exception as e:
            state = self.shared_state[self.camera_id]
            self.shared_state[self.camera_id] = {**state, "error": str(e), "running": False}
            print(f"[{self.camera_id}] Pipeline error: {e}")
        finally:
            state = self.shared_state[self.camera_id]
            self.shared_state[self.camera_id] = {**state, "running": False}
            print(f"[{self.camera_id}] Pipeline stopped. Frames: {frame_count}")