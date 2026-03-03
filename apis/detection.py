"""
apis/detection.py
-----------------
Multi-camera detection resource.
Each camera runs a CameraPipeline in its own process.
Pipeline composition is handled in app.py.
"""

import os
import multiprocessing
from typing import Dict, Optional, Callable

from fastapi import HTTPException
from dotenv import load_dotenv

from apis.base import BaseResource
from schemas import DetectionRequest, DetectionStatus, CameraStatus

load_dotenv()

NUM_CAMERAS = int(os.getenv("NUM_CAMERAS", "1"))
CAMERAS     = {
    f"cam{i}": os.getenv(f"CAMERA_{i}_URL", "")
    for i in range(1, NUM_CAMERAS + 1)
}


class DetectionResource(BaseResource):
    def __init__(self):
        super().__init__()
        self.class_instance = {
            "start"   : self._start,
            "stop"    : self._stop,
            "stop_all": self._stop_all,
        }
        self._manager      = multiprocessing.Manager()
        self._shared_state = self._manager.dict()
        self._processes    : Dict[str, multiprocessing.Process] = {}
        self._stop_events  : Dict[str, multiprocessing.Event]   = {}
        self._pipeline_fn  : Optional[Callable] = None   # set from app.py

    def init(self):
        pass

    def set_pipeline(self, fn: Callable):
        """
        Register the pipeline factory function from app.py.
        fn signature: (camera_id, rtsp_url, shared_state, stop_event) -> None
        """
        self._pipeline_fn = fn

    def on_post(self, req: DetectionRequest):
        return self.get_service(req.action)(req.camera_id)

    def on_get(self):
        return DetectionStatus(cameras={
            cam_id: CameraStatus(**cam_state)
            for cam_id, cam_state in self._shared_state.items()
        })

    # ── Actions ──────────────────────────────

    def _start(self, camera_id: Optional[str] = None):
        if self._pipeline_fn is None:
            raise HTTPException(status_code=500, detail="Pipeline not configured. Call set_pipeline() in app.py.")

        targets = [camera_id] if camera_id else list(CAMERAS.keys())
        started = []

        for cam_id in targets:
            if cam_id not in CAMERAS:
                raise HTTPException(status_code=404, detail=f"Unknown camera '{cam_id}'. Available: {list(CAMERAS.keys())}")
            if cam_id in self._processes and self._processes[cam_id].is_alive():
                raise HTTPException(status_code=409, detail=f"Camera '{cam_id}' already running.")

            stop_event = self._manager.Event()
            process    = multiprocessing.Process(
                target=self._pipeline_fn,
                args=(cam_id, CAMERAS[cam_id], self._shared_state, stop_event),
                daemon=True,
            )
            self._stop_events[cam_id] = stop_event
            self._processes[cam_id]   = process
            process.start()
            started.append(cam_id)

        return {"status": "started", "cameras": started}

    def _stop(self, camera_id: Optional[str] = None):
        targets = [camera_id] if camera_id else list(self._processes.keys())
        stopped = []

        for cam_id in targets:
            if cam_id not in self._processes or not self._processes[cam_id].is_alive():
                raise HTTPException(status_code=409, detail=f"Camera '{cam_id}' is not running.")
            self._stop_events[cam_id].set()
            self._processes[cam_id].join(timeout=5)
            stopped.append(cam_id)

        return {"status": "stopped", "cameras": stopped}

    def _stop_all(self, _=None):
        return self._stop(None)


detection = DetectionResource()