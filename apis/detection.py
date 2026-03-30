"""
apis/detection.py
-----------------
Detection resource — manages camera processes and result streaming.

Routes registered in app.py:
    POST /detection/setup           → configure pipeline services
    POST /detection/start           → start cameras
    POST /detection/stop            → stop cameras
    GET  /detection/status          → current status of all cameras
    GET  /detection/stream/{cam_id} → SSE stream per camera
"""

import multiprocessing
from typing import Dict, Optional, Callable, List

from pydantic import BaseModel

from apis.base import BaseResource
from apis.cameras import camera_registry
from schemas import DetectionStatus, CameraStatus
from error_codes.error_codes import ErrorCode
from error_codes.response import success, error


# ─────────────────────────────────────────────
# Setup schema
# ─────────────────────────────────────────────
class DetectionSetupRequest(BaseModel):
    pipeline: List[str]


# ─────────────────────────────────────────────
# Detection resource
# ─────────────────────────────────────────────
class DetectionResource(BaseResource):
    def __init__(self):
        super().__init__()
        self._manager        = multiprocessing.Manager()
        self._shared_state   = self._manager.dict()
        self._result_queues  : Dict[str, any] = {}
        self._processes      : Dict[str, multiprocessing.Process] = {}
        self._stop_events    : Dict[str, multiprocessing.Event]   = {}
        self._pipeline_fn    : Optional[Callable] = None
        self._pipeline_names : List[str] = []

    def set_pipeline(self, fn: Callable):
        self._pipeline_fn = fn

    # ── Action dispatcher ─────────────────────

    def on_post(self, action: str, camera_id: str = None):
        actions = {
            "start"   : self.on_start,
            "stop"    : self.on_stop,
        }
        if action not in actions:
            return error(ErrorCode.INTERNAL_ERROR, detail=f"Unknown action '{action}'.")
        return actions[action](camera_id)

    # ── Setup ────────────────────────────────

    def on_setup(self, req: DetectionSetupRequest):
        from services import REGISTRY
        unknown = [n for n in req.pipeline if n not in REGISTRY]
        if unknown:
            return error(ErrorCode.UNKNOWN_SERVICE, detail=str(unknown))
        self._pipeline_names = req.pipeline
        return success({
            "status"  : "configured",
            "pipeline": self._pipeline_names,
        })

    # ── Status ───────────────────────────────

    def on_get(self):
        return DetectionStatus(cameras={
            cam_id: CameraStatus(**cam_state)
            for cam_id, cam_state in self._shared_state.items()
        })

    # ── Start ────────────────────────────────

    def on_start(self, camera_id: Optional[str] = None):
        if self._pipeline_fn is None:
            return error(ErrorCode.PIPELINE_NOT_SET)

        if not self._pipeline_names:
            return error(ErrorCode.PIPELINE_NOT_CONFIGURED)

        cameras = camera_registry.all()
        if not cameras:
            return error(ErrorCode.NO_CAMERAS_CONFIGURED)

        targets = [camera_id] if camera_id else list(cameras.keys())
        started = []

        for cam_id in targets:
            if cam_id not in cameras:
                return error(ErrorCode.CAMERA_NOT_FOUND, detail=cam_id)
            if cam_id in self._processes and self._processes[cam_id].is_alive():
                return error(ErrorCode.CAMERA_ALREADY_RUNNING, detail=cam_id)

            self._result_queues[cam_id] = self._manager.Queue()
            stop_event = self._manager.Event()
            process    = multiprocessing.Process(
                target=self._pipeline_fn,
                args=(
                    cam_id,
                    cameras[cam_id],
                    self._shared_state,
                    stop_event,
                    self._result_queues[cam_id],
                    self._pipeline_names,
                ),
                daemon=True,
            )
            self._stop_events[cam_id] = stop_event
            self._processes[cam_id]   = process
            process.start()
            started.append(cam_id)

        return success({
            "status" : "started",
            "cameras": started,
        })

    # ── Stop ─────────────────────────────────

    def on_stop(self, camera_id: Optional[str] = None):
        running = {k: v for k, v in self._processes.items() if v.is_alive()}
        if not running:
            return error(ErrorCode.NO_CAMERAS_RUNNING)

        targets = [camera_id] if camera_id else list(running.keys())
        stopped = []

        for cam_id in targets:
            if cam_id not in self._processes or not self._processes[cam_id].is_alive():
                return error(ErrorCode.NO_CAMERAS_RUNNING, detail=cam_id)
            self._stop_events[cam_id].set()
            self._processes[cam_id].join(timeout=5)
            stopped.append(cam_id)

        return success({
            "status" : "stopped",
            "cameras": stopped,
        })

    # ── Stream ───────────────────────────────

    def get_result_queue(self, cam_id: str):
        if cam_id not in self._result_queues:
            return None, ErrorCode.STREAM_CAMERA_NOT_FOUND
        return self._result_queues[cam_id], None


# ── Singleton ─────────────────────────────────
detection = DetectionResource()