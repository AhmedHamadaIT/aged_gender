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

import logging
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
from typing import Dict, List, Optional

from fastapi import HTTPException

from apis.base import BaseResource
from apis.cameras import camera_registry
from apis.tasks import task_registry
from schemas import DetectionRequest, DetectionStatus, CameraStatus

log = logging.getLogger(__name__)


def _detection_http_error(status_code: int, detail: str) -> None:
    """Log client errors so server logs show the same message as the JSON ``detail`` field."""
    log.warning("[detection] HTTP %s: %s", status_code, detail)
    raise HTTPException(status_code=status_code, detail=detail)


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
        self._embedding_stop      = None
        self._frame_seq           : Dict[str, object] = {}
        self._frame_seq_locks     : Dict[str, object] = {}
        self._bus_fatal_events    : Dict[str, object] = {}
        self._event_seq           : Dict[str, object] = {}
        self._event_seq_locks     : Dict[str, object] = {}
        self._watchdog_respawn_times: Dict[str, List[float]] = defaultdict(list)
        self._task_queues_ref: Dict[str, Dict[str, object]] = {}

    # ── Status ───────────────────────────────────────────────────────────────

    def on_post(self, req: DetectionRequest):
        if req.action == "start":
            return self._start(req.camera_id, all_channels=req.all_channels)
        if req.action == "stop":
            return self._stop(req.camera_id)
        if req.action == "stop_all":
            return self._stop_all()
        _detection_http_error(
            400,
            f"Unknown action '{req.action}'. Available: start, stop, stop_all",
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

    def _watchdog_tick(self) -> None:
        """Called periodically when WATCHDOG_ENABLED (from asyncio watchdog task)."""
        if os.getenv("WATCHDOG_ENABLED", "false").lower() not in ("true", "1", "yes"):
            return
        for cam_id, proc in list(self._bus_processes.items()):
            ev = self._stop_events.get(cam_id)
            if ev is not None and ev.is_set():
                continue
            fatal = self._bus_fatal_events.get(cam_id)
            dead = proc is None or not proc.is_alive()
            if fatal is not None and fatal.is_set():
                dead = True
            if not dead:
                continue
            st = dict(self._shared_state.get(cam_id, {}))
            if st.get("stopped_reason") == "stream_exhausted":
                continue
            if not st.get("running") and not (fatal is not None and fatal.is_set()):
                continue
            try:
                self._restart_channel(str(cam_id))
            except Exception:
                import logging

                logging.getLogger(__name__).exception(
                    "watchdog: restart failed for %s", cam_id
                )

    def _restart_channel(self, cam_id: str) -> None:
        """Tear down and respawn FrameBus + task workers for one camera."""
        from task_worker import run_task_worker

        window = float(os.getenv("WATCHDOG_RESPAWN_WINDOW_SEC", "600"))
        cap = int(os.getenv("WATCHDOG_MAX_RESPAWNS", "5"))
        now = time.time()
        times = [t for t in self._watchdog_respawn_times[cam_id] if now - t <= window]
        if len(times) >= cap:
            row = dict(self._shared_state.get(cam_id, {}))
            row["error"] = row.get("error") or "watchdog_max_respawns"
            row["running"] = False
            row["stopped_reason"] = "watchdog_max_respawns"
            row["state_updated_at"] = time.time()
            self._shared_state[cam_id] = row
            return
        times.append(now)
        self._watchdog_respawn_times[cam_id] = times

        import logging

        logging.getLogger(__name__).warning(
            "resilience_watchdog: restarting channel %s (respawn #%d in window)",
            cam_id,
            len(times),
        )

        old_stop = self._stop_events.get(cam_id)
        if old_stop is not None:
            old_stop.set()

        proc = self._bus_processes.get(cam_id)
        if proc is not None and proc.is_alive():
            proc.join(timeout=2)
            if proc.is_alive():
                proc.terminate()
        for p in list(self._task_processes.get(cam_id, {}).values()):
            if p.is_alive():
                p.join(timeout=2)
                if p.is_alive():
                    p.terminate()

        cameras = camera_registry.all()
        chan_tasks = [
            t for t in task_registry.get_enabled() if str(t["channelId"]) == cam_id
        ]
        if not chan_tasks or cam_id not in cameras:
            return

        if cam_id not in self._frame_seq:
            self._frame_seq[cam_id] = self._manager.Value("Q", 0)
            self._frame_seq_locks[cam_id] = self._manager.Lock()
        if cam_id not in self._bus_fatal_events:
            self._bus_fatal_events[cam_id] = self._manager.Event()
        if cam_id not in self._event_seq:
            self._event_seq[cam_id] = self._manager.Value("Q", 0)
            self._event_seq_locks[cam_id] = self._manager.Lock()

        try:
            self._bus_fatal_events[cam_id].clear()
        except Exception:
            pass

        stop_event = self._manager.Event()
        task_queues: Dict[str, object] = {}
        _tq_max = max(1, int(os.getenv("TASK_QUEUE_MAXSIZE", "256")))
        self._task_processes[cam_id] = {}

        for task_cfg in chan_tasks:
            task_id = str(task_cfg["taskId"])
            q = self._manager.Queue(maxsize=_tq_max)
            task_queues[task_id] = q
            p = multiprocessing.Process(
                target=run_task_worker,
                args=(
                    cam_id,
                    task_cfg,
                    q,
                    self._result_queue,
                    stop_event,
                    self._shared_state,
                    self._event_seq[cam_id],
                    self._event_seq_locks[cam_id],
                ),
                daemon=True,
            )
            self._task_processes[cam_id][task_id] = p
            p.start()

        bus = multiprocessing.Process(
            target=_run_frame_bus,
            args=(
                cam_id,
                cameras[cam_id],
                self._shared_state,
                stop_event,
                task_queues,
                self._embedding_queue,
                self._frame_seq[cam_id],
                self._frame_seq_locks[cam_id],
                self._bus_fatal_events[cam_id],
            ),
            daemon=True,
        )
        self._bus_processes[cam_id] = bus
        self._stop_events[cam_id] = stop_event
        self._task_queues_ref[cam_id] = task_queues
        row = dict(self._shared_state.get(cam_id, {}))
        row["respawn_count"] = int(row.get("respawn_count", 0)) + 1
        row["running"] = True
        row["error"] = None
        row["stopped_reason"] = None
        row["state_updated_at"] = time.time()
        self._shared_state[cam_id] = row
        bus.start()

    # ── Start ─────────────────────────────────────────────────────────────────

    def _start(self, camera_id: Optional[str] = None, all_channels: bool = False):
        from task_worker import run_task_worker

        all_enabled = task_registry.get_enabled()
        tasks   = all_enabled
        cameras = camera_registry.all()

        if not tasks:
            _detection_http_error(
                400,
                "No enabled tasks configured. Call POST /api/tasks first.",
            )
        if not cameras:
            _detection_http_error(
                400,
                "No cameras configured. Call POST /cameras first.",
            )

        # Optionally filter to a single camera
        if camera_id:
            tasks = [t for t in tasks if str(t["channelId"]) == str(camera_id)]
            if not tasks:
                chans = sorted({str(t["channelId"]) for t in all_enabled})
                _detection_http_error(
                    404,
                    (
                        f"No enabled tasks found for camera '{camera_id}'. "
                        f"Enabled task channelIds: {chans}. "
                        f"Registered camera ids: {sorted(cameras.keys())}."
                    ),
                )
        else:
            distinct = sorted({str(t["channelId"]) for t in tasks})
            if len(distinct) > 1 and not all_channels:
                _detection_http_error(
                    400,
                    (
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
                _detection_http_error(
                    404,
                    f"No camera registered for channelId '{chan_id}'. "
                    f"Register it via POST /cameras with id='{chan_id}'.",
                )
            if chan_id in self._bus_processes and self._bus_processes[chan_id].is_alive():
                _detection_http_error(
                    409,
                    f"Camera '{chan_id}' is already running.",
                )

            if chan_id not in self._frame_seq:
                self._frame_seq[chan_id] = self._manager.Value("Q", 0)
                self._frame_seq_locks[chan_id] = self._manager.Lock()
            if chan_id not in self._bus_fatal_events:
                self._bus_fatal_events[chan_id] = self._manager.Event()
            if chan_id not in self._event_seq:
                self._event_seq[chan_id] = self._manager.Value("Q", 0)
                self._event_seq_locks[chan_id] = self._manager.Lock()

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
                    args=(
                        chan_id,
                        task_cfg,
                        q,
                        self._result_queue,
                        stop_event,
                        self._shared_state,
                        self._event_seq[chan_id],
                        self._event_seq_locks[chan_id],
                    ),
                    daemon=True,
                )
                self._task_processes[chan_id][task_id] = p
                p.start()
                started_tasks.append(task_id)

            # One FrameBus per camera — fans frames out to all task queues
            bus = multiprocessing.Process(
                target=_run_frame_bus,
                args=(
                    chan_id,
                    cameras[chan_id],
                    self._shared_state,
                    stop_event,
                    task_queues,
                    self._embedding_queue,
                    self._frame_seq[chan_id],
                    self._frame_seq_locks[chan_id],
                    self._bus_fatal_events[chan_id],
                ),
                daemon=True,
            )
            self._bus_processes[chan_id] = bus
            self._stop_events[chan_id]   = stop_event
            self._task_queues_ref[chan_id] = task_queues
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
            _detection_http_error(409, "No cameras are currently running.")

        targets = [camera_id] if camera_id else list(running.keys())
        stopped = []

        drain_timeout = float(os.getenv("STOP_DRAIN_TIMEOUT_SEC", "3"))
        for cam_id in targets:
            if cam_id not in self._bus_processes or not self._bus_processes[cam_id].is_alive():
                _detection_http_error(409, f"Camera '{cam_id}' is not running.")
            self._stop_events[cam_id].set()
            deadline = time.time() + drain_timeout
            tq = self._task_queues_ref.get(cam_id, {})
            while time.time() < deadline and tq:
                try:
                    if all(q.empty() for q in tq.values()):  # type: ignore[attr-defined]
                        break
                except Exception:
                    break
                time.sleep(0.05)
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
def _run_frame_bus(
    camera_id,
    rtsp_url,
    shared_state,
    stop_event,
    task_queues,
    embedding_queue=None,
    frame_seq=None,
    frame_seq_lock=None,
    bus_fatal_event=None,
):
    from frame_bus import FrameBus

    FrameBus(
        camera_id,
        rtsp_url,
        shared_state,
        stop_event,
        task_queues,
        embedding_queue,
        frame_seq_counter=frame_seq,
        frame_seq_lock=frame_seq_lock,
        bus_fatal_event=bus_fatal_event,
    ).run()


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
