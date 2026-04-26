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
import os
import time

# Parent process loads CUDA-backed models (e.g. ReID/OSNet) before workers start.
# Linux default start method is "fork"; forked children cannot re-init CUDA.
# "spawn" starts a fresh interpreter per worker (see PyTorch / Ultralytics docs).
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

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
        self._embedding_queue = self._manager.Queue(maxsize=200)

        # Keyed by camera_id (str of channelId)
        self._bus_processes       : Dict[str, multiprocessing.Process]            = {}
        self._task_processes      : Dict[str, Dict[str, multiprocessing.Process]] = {}
        self._stop_events         : Dict[str, object]                             = {}
        self._embedding_worker    : multiprocessing.Process = None

    # ── Status ───────────────────────────────────────────────────────────────

    def on_post(self, req: DetectionRequest):
        if req.action == "start":
            return self._start(req.camera_id, all_channels=req.all_channels)
        if req.action == "stop":
            return self._stop(req.camera_id)
        if req.action == "stop_all":
            return self._stop_all()
        raise HTTPException(
            status_code=400,
            detail=f"Unknown action '{req.action}'. Available: start, stop, stop_all",
        )

    def enrich_shared_camera_row(self, cam_id: str, cam_state: dict) -> dict:
        """Merge manager shared_state with parent process liveness; used by /detection/status and /stream/metrics."""
        out = dict(cam_state)
        proc = self._bus_processes.get(cam_id)
        if proc is not None:
            alive = proc.is_alive()
            out["framebus_process_alive"] = alive
            if out.get("running") and not alive:
                out["running"] = False
                if not out.get("error"):
                    out["stopped_reason"] = out.get("stopped_reason") or "framebus_process_exited"
        else:
            out["framebus_process_alive"] = None

        ts = out.get("state_updated_at")
        if ts is not None:
            try:
                out["last_state_update_age_sec"] = max(0.0, time.time() - float(ts))
            except (TypeError, ValueError):
                out["last_state_update_age_sec"] = None
        return out

    def on_get(self):
        return DetectionStatus(
            cameras={
                cam_id: CameraStatus(
                    **self.enrich_shared_camera_row(cam_id, dict(cam_state))
                )
                for cam_id, cam_state in self._shared_state.items()
            }
        )

    # ── Start ─────────────────────────────────────────────────────────────────

    def _start(self, camera_id: Optional[str] = None, all_channels: bool = False):
        from task_worker import run_task_worker

        all_enabled = task_registry.get_enabled()
        tasks   = all_enabled
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
                chans = sorted({str(t["channelId"]) for t in all_enabled})
                raise HTTPException(
                    status_code=404,
                    detail=(
                        f"No enabled tasks found for camera '{camera_id}'. "
                        f"Enabled task channelIds: {chans}. "
                        f"Registered camera ids: {sorted(cameras.keys())}."
                    ),
                )
        else:
            distinct = sorted({str(t["channelId"]) for t in tasks})
            if len(distinct) > 1 and not all_channels:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Multiple enabled task channels: {distinct}. "
                        "Use POST /detection/start?camera_id=<id> to start one channel, "
                        "or pass all_channels=true to start all."
                    ),
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
            # Bounded queue between FrameBus and each task worker. Too small + slow
            # startup (TRT, 4K decode) causes put_nowait drops; override via TASK_QUEUE_MAXSIZE.
            _tq_max = max(1, int(os.getenv("TASK_QUEUE_MAXSIZE", "256")))
            for task_cfg in chan_tasks:
                task_id = str(task_cfg["taskId"])
                q = self._manager.Queue(maxsize=_tq_max)
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
                args=(chan_id, cameras[chan_id], self._shared_state, stop_event, task_queues, self._embedding_queue),
                daemon=True,
            )
            self._bus_processes[chan_id] = bus
            self._stop_events[chan_id]   = stop_event
            bus.start()
            started_cameras.append(chan_id)

        # ── Start the shared EmbeddingWorker (if not already running) ─────
        if self._embedding_worker is None or not self._embedding_worker.is_alive():
            self._embedding_stop = self._manager.Event()
            self._embedding_worker = multiprocessing.Process(
                target=_run_embedding_worker,
                args=(self._embedding_queue, self._embedding_stop),
                daemon=True,
            )
            self._embedding_worker.start()

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

        # ── Stop EmbeddingWorker if no cameras remain running ─────────────
        still_running = {k: v for k, v in self._bus_processes.items() if v.is_alive()}
        if not still_running and self._embedding_worker and self._embedding_worker.is_alive():
            self._embedding_stop.set()
            self._embedding_worker.join(timeout=10)

        return {"status": "stopped", "cameras": stopped}

    def _stop_all(self, _=None):
        return self._stop(None)

    # ── SSE ───────────────────────────────────────────────────────────────────

    def result_queue(self):
        return self._result_queue


# ─────────────────────────────────────────────
# Top-level picklable entry for the FrameBus process
# ─────────────────────────────────────────────
def _run_frame_bus(camera_id, rtsp_url, shared_state, stop_event, task_queues, embedding_queue=None):
    from frame_bus import FrameBus
    FrameBus(camera_id, rtsp_url, shared_state, stop_event, task_queues, embedding_queue).run()


def _run_embedding_worker(embedding_queue, stop_event):
    from embedding_worker import run_embedding_worker
    run_embedding_worker(embedding_queue, stop_event)


# ── Singleton ─────────────────────────────────
# Spawned FrameBus / task_worker processes re-import this module. Creating
# ``Manager()`` at import time in a child triggers:
#   RuntimeError: ... start a new process before ... bootstrapping phase
# Only the uvicorn process (MainProcess) owns the shared manager and queues.
if multiprocessing.current_process().name == "MainProcess":
    detection = DetectionResource()
else:
    detection = None  # workers only need picklable targets above, not this API handle
