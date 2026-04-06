"""
apis/detection.py
-----------------
Detection resource — manages camera processes and result streaming.

Cameras and tasks are configured separately before starting:
  - POST /cameras          → register cameras (camera_id → rtsp_url)
  - POST /api/tasks        → register tasks (algorithmType, channelId, config)

On start, one FrameBus process is spawned per unique channelId, and one
task worker process is spawned per enabled task. Tasks that share a camera
share the same FrameBus — the bus fans frames out to each task's queue.

Routes registered in app.py:
    POST /detection/start   → start all (or one) camera
    POST /detection/stop    → stop all (or one) camera
    GET  /detection/status  → current status of all cameras
    GET  /detection/stream  → SSE stream of crossing events
"""

import multiprocessing
from collections import defaultdict
from typing import Dict, Optional

from fastapi import HTTPException

from apis.base import BaseResource
from apis.cameras import camera_registry
from apis.tasks import task_registry
from schemas import DetectionRequest, DetectionStatus, CameraStatus


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
        self._result_queue   = self._manager.Queue()

        # Keyed by camera_id (str of channelId)
        self._bus_processes  : Dict[str, multiprocessing.Process]            = {}
        self._task_processes : Dict[str, Dict[str, multiprocessing.Process]] = {}
        self._stop_events    : Dict[str, object]                             = {}

    # ── Status ───────────────────────────────────────────────────────────────

    def on_post(self, req: DetectionRequest):
        return self.get_service(req.action)(req.camera_id)

    def on_get(self):
        return DetectionStatus(cameras={
            cam_id: CameraStatus(**cam_state)
            for cam_id, cam_state in self._shared_state.items()
        })

    # ── Start ─────────────────────────────────────────────────────────────────

    def _start(self, camera_id: Optional[str] = None):
        from task_worker import run_task_worker

        tasks   = task_registry.get_enabled()
        cameras = camera_registry.all()

        if not tasks:
            raise HTTPException(
                status_code=400,
                detail="No enabled tasks configured. Call POST /api/tasks first."
            )
        if not cameras:
            raise HTTPException(
                status_code=400,
                detail="No cameras configured. Call POST /cameras first."
            )

        # Optionally filter to a single camera
        if camera_id:
            tasks = [t for t in tasks if str(t["channelId"]) == str(camera_id)]
            if not tasks:
                raise HTTPException(
                    status_code=404,
                    detail=f"No enabled tasks found for camera '{camera_id}'."
                )

        # Group tasks by channelId — one FrameBus per camera
        channel_tasks: Dict[str, list] = defaultdict(list)
        for task in tasks:
            channel_tasks[str(task["channelId"])].append(task)

        started_cameras = []
        started_tasks   = []

        for chan_id, chan_tasks in channel_tasks.items():
            if chan_id not in cameras:
                raise HTTPException(
                    status_code=404,
                    detail=f"No camera registered for channelId '{chan_id}'. "
                           f"Register it via POST /cameras with id='{chan_id}'."
                )
            if chan_id in self._bus_processes and self._bus_processes[chan_id].is_alive():
                raise HTTPException(
                    status_code=409,
                    detail=f"Camera '{chan_id}' is already running."
                )

            stop_event  = self._manager.Event()
            task_queues = {}

            # One task worker process per task
            self._task_processes.setdefault(chan_id, {})
            for task_cfg in chan_tasks:
                task_id = str(task_cfg["taskId"])
                q = self._manager.Queue(maxsize=10)
                task_queues[task_id] = q

                p = multiprocessing.Process(
                    target=run_task_worker,
                    args=(chan_id, task_cfg, q, self._result_queue, stop_event),
                    daemon=True,
                )
                self._task_processes[chan_id][task_id] = p
                p.start()
                started_tasks.append(task_id)

            # One FrameBus per camera — fans frames out to all task queues
            bus = multiprocessing.Process(
                target=_run_frame_bus,
                args=(chan_id, cameras[chan_id], self._shared_state, stop_event, task_queues),
                daemon=True,
            )
            self._bus_processes[chan_id] = bus
            self._stop_events[chan_id]   = stop_event
            bus.start()
            started_cameras.append(chan_id)

        return {
            "status" : "started",
            "cameras": started_cameras,
            "tasks"  : started_tasks,
        }

    # ── Stop ──────────────────────────────────────────────────────────────────

    def _stop(self, camera_id: Optional[str] = None):
        running = {k: v for k, v in self._bus_processes.items() if v.is_alive()}
        if not running:
            raise HTTPException(status_code=409, detail="No cameras are currently running.")

        targets = [camera_id] if camera_id else list(running.keys())
        stopped = []

        for cam_id in targets:
            if cam_id not in self._bus_processes or not self._bus_processes[cam_id].is_alive():
                raise HTTPException(status_code=409, detail=f"Camera '{cam_id}' is not running.")
            self._stop_events[cam_id].set()
            self._bus_processes[cam_id].join(timeout=5)
            for p in self._task_processes.get(cam_id, {}).values():
                p.join(timeout=5)
            stopped.append(cam_id)

        return {"status": "stopped", "cameras": stopped}

    def _stop_all(self, _=None):
        return self._stop(None)

    # ── SSE ───────────────────────────────────────────────────────────────────

    def result_queue(self):
        return self._result_queue


# ─────────────────────────────────────────────
# Top-level picklable entry for the FrameBus process
# ─────────────────────────────────────────────
def _run_frame_bus(camera_id, rtsp_url, shared_state, stop_event, task_queues):
    from frame_bus import FrameBus
    FrameBus(camera_id, rtsp_url, shared_state, stop_event, task_queues).run()


# ── Singleton ─────────────────────────────────
detection = DetectionResource()
