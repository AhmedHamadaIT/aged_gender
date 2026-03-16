"""
apis/detection.py
-----------------
Detection resource — manages camera processes and result streaming.

Routes registered in app.py:
    POST /detection/setup   → configure pipeline services
    POST /detection/start   → start cameras
    POST /detection/stop    → stop cameras
    GET  /detection/status  → current status of all cameras
    GET  /detection/stream  → SSE stream of frame results
"""

import multiprocessing
import queue
from typing import Dict, Optional, Callable, List

from fastapi import HTTPException
from pydantic import BaseModel

from apis.base import BaseResource
from apis.cameras import camera_registry
from schemas import DetectionRequest, DetectionStatus, CameraStatus


# ─────────────────────────────────────────────
# Setup schema
# ─────────────────────────────────────────────
class DetectionSetupRequest(BaseModel):
    pipeline: List[str]   # e.g. ["detector", "age_gender"]


# ─────────────────────────────────────────────
# Detection resource
# ─────────────────────────────────────────────
class DetectionResource(BaseResource):
    def __init__(self):
        super().__init__()
        self.class_instance = {
            "start"   : self._start,
            "stop"    : self._stop,
            "stop_all": self._stop_all,
        }
        self._manager        = multiprocessing.Manager()
        self._shared_state   = self._manager.dict()
        self._result_queue   = self._manager.Queue()   # frames + results from all cameras
        self._processes      : Dict[str, multiprocessing.Process] = {}
        self._stop_events    : Dict[str, multiprocessing.Event]   = {}
        self._pipeline_fn    : Optional[Callable] = None
        self._pipeline_names : List[str] = []

    def set_pipeline(self, fn: Callable):
        """Set the pipeline factory function from app.py."""
        self._pipeline_fn = fn

    # ── Setup ────────────────────────────────

    def on_setup(self, req: DetectionSetupRequest):
        """Configure which services run in the pipeline."""
        from services import REGISTRY
        unknown = [n for n in req.pipeline if n not in REGISTRY]
        if unknown:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown services: {unknown}. Available: {list(REGISTRY.keys())}"
            )
        self._pipeline_names = req.pipeline
        return {
            "status"  : "configured",
            "pipeline": self._pipeline_names,
        }

    # ── Start / Stop ─────────────────────────

    def on_post(self, req: DetectionRequest):
        return self.get_service(req.action)(req.camera_id)

    def on_get(self):
        return DetectionStatus(cameras={
            cam_id: CameraStatus(**cam_state)
            for cam_id, cam_state in self._shared_state.items()
        })

    def _start(self, camera_id: Optional[str] = None):
        if self._pipeline_fn is None:
            raise HTTPException(status_code=500, detail="Pipeline not configured.")
        if not self._pipeline_names:
            raise HTTPException(status_code=400, detail="No pipeline configured. Call POST /detection/setup first.")

        cameras = camera_registry.all()
        if not cameras:
            raise HTTPException(status_code=400, detail="No cameras configured. Call POST /cameras first.")

        targets = [camera_id] if camera_id else list(cameras.keys())
        started = []

        for cam_id in targets:
            if cam_id not in cameras:
                raise HTTPException(
                    status_code=404,
                    detail=f"Camera '{cam_id}' not found. Available: {list(cameras.keys())}"
                )
            if cam_id in self._processes and self._processes[cam_id].is_alive():
                raise HTTPException(status_code=409, detail=f"Camera '{cam_id}' already running.")

            stop_event = self._manager.Event()
            process    = multiprocessing.Process(
                target=self._pipeline_fn,
                args=(
                    cam_id,
                    cameras[cam_id],
                    self._shared_state,
                    stop_event,
                    self._result_queue,
                    self._pipeline_names,
                ),
                daemon=True,
            )
            self._stop_events[cam_id] = stop_event
            self._processes[cam_id]   = process
            process.start()
            started.append(cam_id)

        return {"status": "started", "cameras": started}

    def _stop(self, camera_id: Optional[str] = None):
        running = {k: v for k, v in self._processes.items() if v.is_alive()}
        if not running:
            raise HTTPException(status_code=409, detail="No cameras are currently running.")

        targets = [camera_id] if camera_id else list(running.keys())
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

    # ── SSE stream ───────────────────────────

    def result_queue(self):
        return self._result_queue


# ── Singleton ─────────────────────────────────
detection = DetectionResource()