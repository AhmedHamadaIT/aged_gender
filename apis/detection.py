"""
apis/detection.py
-----------------
Detection resource — manages camera processes and result streaming.

Cameras and tasks are configured separately before starting:
  - POST /cameras          → register cameras (camera_id → rtsp_url)
  - POST /api/tasks        → register tasks (algorithmType, channelId, config)

On start, a strict 10-step validation gate runs before ANY subprocess is
spawned.  Only after all steps pass are processes created and verified:

  Steps 1–7  : StreamValidator (in-process, synchronous)
  Step  8    : FrameBus subprocess signals model-ready within timeout
  Step  9    : Each task worker subprocess signals ready within timeout
  Step 10    : Annotation / frame processing starts

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
from utils.live_stream_overlay import build_live_stream_overlay
from utils.error_codes import (
    ValidationFailure,
    BUS_INIT_TIMEOUT,
    WORKER_INIT_TIMEOUT,
    MODEL_INIT_FAILED,
    WORKER_INIT_FAILED,
)

log = logging.getLogger(__name__)

# Init-phase timeouts — how long the parent waits for subprocess ready signals.
_WORKER_INIT_TIMEOUT_SEC = float(os.getenv("WORKER_INIT_TIMEOUT_SEC", "15.0"))
_BUS_INIT_TIMEOUT_SEC    = float(os.getenv("BUS_INIT_TIMEOUT_SEC",    "30.0"))


def _detection_http_error(status_code: int, detail: str) -> None:
    """Log client errors so server logs show the same message as the JSON ``detail`` field."""
    log.warning("[detection] HTTP %s: %s", status_code, detail)
    raise HTTPException(status_code=status_code, detail=detail)


def _validation_failure_response(failure: ValidationFailure) -> None:
    """Raise 422 Unprocessable Entity carrying the structured ValidationFailure payload."""
    log.warning(
        "[detection] validation_failure stage=%d code=%s cam=%s task=%s: %s",
        failure.stage,
        failure.error_code,
        failure.camera_id,
        failure.task_id,
        failure.message,
    )
    raise HTTPException(status_code=422, detail=failure.to_dict())


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

        # task_validity_map: {task_id (str) → {"enabled": bool, "exists": bool}}
        # Written by API handlers (update_task_validity) when tasks are modified.
        # Read by task_worker processes every TASK_VALIDITY_POLL_SEC to detect
        # runtime task removal / disable without full watchdog respawn.
        self._task_validity_map: object = self._manager.dict()

        # Keyed by camera_id (str of channelId)
        self._bus_processes       : Dict[str, multiprocessing.Process]            = {}
        self._task_processes      : Dict[str, Dict[str, multiprocessing.Process]] = {}
        self._stop_events         : Dict[str, object]                             = {}
        self._embedding_worker    : multiprocessing.Process = None
        self._embedding_stop      = None
        self._frame_seq           : Dict[str, object] = {}
        self._frame_seq_locks     : Dict[str, object] = {}
        self._bus_fatal_events    : Dict[str, object] = {}
        self._bus_ready_events    : Dict[str, object] = {}
        self._worker_ready_events : Dict[str, Dict[str, object]] = {}
        self._event_seq           : Dict[str, object] = {}
        self._event_seq_locks     : Dict[str, object] = {}
        self._watchdog_respawn_times: Dict[str, List[float]] = defaultdict(list)
        self._task_queues_ref: Dict[str, Dict[str, object]] = {}
        # M-5: live overlay store — camera_id → overlay dict built from enabled tasks.
        self._live_overlay_store: Dict[str, dict] = {}

    # ── Task validity map ─────────────────────────────────────────────────────

    def update_task_validity(
        self, task_id: str, *, enabled: bool, exists: bool
    ) -> None:
        """
        Called by task API handlers (PUT / DELETE) so running task workers are
        notified of state changes without requiring a full watchdog respawn.
        Thread-safe: writes go through the multiprocessing Manager.
        """
        try:
            self._task_validity_map[task_id] = {"enabled": enabled, "exists": exists}
            log.info(
                "[detection] task_validity updated: task_id=%s enabled=%s exists=%s",
                task_id, enabled, exists,
            )
        except Exception:
            log.exception("[detection] update_task_validity failed for task_id=%s", task_id)

    def remove_task_validity(self, task_id: str) -> None:
        """Mark task as removed — workers will detect this and stop cleanly."""
        self.update_task_validity(task_id, enabled=False, exists=False)

    # ── M-5: Cross-line hot-reload ────────────────────────────────────────────

    def _set_channel_live_overlay(self, cam_id: str, tasks: list) -> None:
        """Rebuild and store the live-stream overlay for a camera."""
        overlay = build_live_stream_overlay(tasks)
        self._live_overlay_store[cam_id] = overlay

    def reload_cross_line_task(self, task_id: int, task: dict) -> dict:
        """
        Hot-reload a CROSS_LINE task worker:
        1. Refresh the live-stream overlay for the owning camera.
        2. Terminate the old task worker process and start a fresh one.

        Returns {"applied": bool, "reason": str?, "overlay_refreshed": bool}.
        """
        cam_id = str(task.get("channelId", ""))
        tid_str = str(task_id)

        bus_proc = self._bus_processes.get(cam_id)
        if bus_proc is None or not bus_proc.is_alive():
            return {"applied": False, "reason": "camera_not_running"}

        # Refresh overlay from all currently-enabled tasks on this camera.
        try:
            chan_tasks = [
                t for t in task_registry.get_enabled()
                if str(t.get("channelId", "")) == cam_id
            ]
            self._set_channel_live_overlay(cam_id, chan_tasks)
            overlay_refreshed = True
        except Exception:
            overlay_refreshed = False

        # Respawn the task worker.
        old_proc = (self._task_processes.get(cam_id) or {}).get(tid_str)
        if old_proc is not None and old_proc.is_alive():
            old_proc.terminate()
            try:
                old_proc.join(timeout=3.0)
            except Exception:
                pass

        q = (self._task_queues_ref.get(cam_id) or {}).get(tid_str)
        stop_event = self._stop_events.get(cam_id)
        if q is None or stop_event is None:
            return {
                "applied": False,
                "reason": "no_queue_or_stop_event",
                "overlay_refreshed": overlay_refreshed,
            }

        from task_worker import run_task_worker  # noqa: PLC0415

        worker_ready = self._worker_ready_events.get(cam_id, {}).get(tid_str)
        if worker_ready is None:
            worker_ready = self._manager.Event()
            self._worker_ready_events.setdefault(cam_id, {})[tid_str] = worker_ready
        else:
            worker_ready.clear()

        new_proc = multiprocessing.Process(
            target=run_task_worker,
            args=(
                cam_id,
                task,
                q,
                self._result_queue,
                stop_event,
                self._shared_state,
                self._event_seq.get(cam_id),
                self._event_seq_locks.get(cam_id),
            ),
            kwargs={"worker_ready_event": worker_ready, "task_validity_map": self._task_validity_map},
            daemon=True,
        )
        new_proc.start()
        self._task_processes.setdefault(cam_id, {})[tid_str] = new_proc

        # Wait briefly for worker to signal readiness.
        worker_ready.wait(timeout=5.0)

        log.info(
            "[detection] CROSS_LINE task %s reloaded for cam %s (pid=%s)",
            task_id, cam_id, new_proc.pid,
        )
        return {"applied": True, "overlay_refreshed": overlay_refreshed}

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
                log.exception("watchdog: restart failed for %s", cam_id)

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

        log.warning(
            "resilience_watchdog: restarting channel %s (respawn #%d in window)",
            cam_id, len(times),
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

        self._ensure_channel_ipc(cam_id)

        stop_event = self._manager.Event()
        task_queues: Dict[str, object] = {}
        _tq_max = max(1, int(os.getenv("TASK_QUEUE_MAXSIZE", "256")))
        self._task_processes[cam_id] = {}
        self._worker_ready_events[cam_id] = {}

        for task_cfg in chan_tasks:
            task_id = str(task_cfg["taskId"])
            q = self._manager.Queue(maxsize=_tq_max)
            task_queues[task_id] = q
            worker_ready = self._manager.Event()
            self._worker_ready_events[cam_id][task_id] = worker_ready
            # Refresh validity map
            self._task_validity_map[task_id] = {
                "enabled": bool(task_cfg.get("enable", True)),
                "exists": True,
            }
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
                kwargs={
                    "worker_ready_event": worker_ready,
                    "task_validity_map": self._task_validity_map,
                },
                daemon=True,
            )
            self._task_processes[cam_id][task_id] = p
            p.start()

        bus_ready = self._manager.Event()
        self._bus_ready_events[cam_id] = bus_ready

        live_overlay = build_live_stream_overlay(chan_tasks)
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
                live_overlay,
            ),
            kwargs={"bus_ready_event": bus_ready, "cpu_affinity": _cpu_affinity_for(cam_id)},
            daemon=True,
        )
        self._bus_processes[cam_id] = bus
        self._stop_events[cam_id]   = stop_event
        self._task_queues_ref[cam_id] = task_queues

        row = dict(self._shared_state.get(cam_id, {}))
        row["respawn_count"] = int(row.get("respawn_count", 0)) + 1
        row["running"] = True
        row["error"] = None
        row["stopped_reason"] = None
        row["state_updated_at"] = time.time()
        self._shared_state[cam_id] = row
        bus.start()

    def _ensure_channel_ipc(self, cam_id: str) -> None:
        """Create Manager-backed IPC objects for a channel if not already present."""
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

    # ── Start ─────────────────────────────────────────────────────────────────

    def _start(self, camera_id: Optional[str] = None, all_channels: bool = False):
        from task_worker import run_task_worker
        from services.stream_validator import StreamValidator

        validator = StreamValidator(camera_registry, task_registry)
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

        # ── Step 1–7: Run StreamValidator for EVERY channel before spawning anything ──
        for chan_id, chan_tasks in channel_tasks.items():
            rtsp_url = cameras.get(chan_id, "")
            failure = validator.validate_channel(chan_id, rtsp_url, chan_tasks)
            if failure:
                _validation_failure_response(failure)

        started_cameras = []
        started_tasks   = []

        self._reap_finished_processes()

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

            self._ensure_channel_ipc(chan_id)

            stop_event  = self._manager.Event()
            task_queues = {}

            self._task_processes.setdefault(chan_id, {})
            self._worker_ready_events.setdefault(chan_id, {})
            _tq_max = max(1, int(os.getenv("TASK_QUEUE_MAXSIZE", "256")))

            # ── Spawn task workers first (they idle on empty queue) ────────
            task_init_start = time.monotonic()
            for task_cfg in chan_tasks:
                task_id = str(task_cfg["taskId"])
                q = self._manager.Queue(maxsize=_tq_max)
                task_queues[task_id] = q

                worker_ready = self._manager.Event()
                self._worker_ready_events[chan_id][task_id] = worker_ready

                # Populate validity map so the worker can poll immediately
                self._task_validity_map[task_id] = {
                    "enabled": bool(task_cfg.get("enable", True)),
                    "exists": True,
                }

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
                    kwargs={
                        "worker_ready_event": worker_ready,
                        "task_validity_map": self._task_validity_map,
                    },
                    daemon=True,
                )
                self._task_processes[chan_id][task_id] = p
                p.start()
                started_tasks.append(task_id)

            # ── Step 9: Wait for all task workers to signal ready ──────────
            for task_cfg in chan_tasks:
                task_id = str(task_cfg["taskId"])
                ready_ev = self._worker_ready_events[chan_id][task_id]
                algorithm = task_cfg.get("algorithmType", "")
                signalled = ready_ev.wait(timeout=_WORKER_INIT_TIMEOUT_SEC)
                if not signalled:
                    log.error(
                        "[detection] Worker init timeout: cam=%s task=%s algo=%s "
                        "timeout=%.1fs — killing spawned processes",
                        chan_id, task_id, algorithm, _WORKER_INIT_TIMEOUT_SEC,
                    )
                    self._kill_channel_processes(chan_id, stop_event)
                    st = dict(self._shared_state.get(chan_id, {}))
                    init_error = st.get("worker_init_error") or (
                        f"Worker for task {task_id} ({algorithm}) did not signal "
                        f"ready within {_WORKER_INIT_TIMEOUT_SEC:.0f}s"
                    )
                    failure = ValidationFailure(
                        stage=9,
                        stage_name="worker_initialized",
                        error_code=WORKER_INIT_TIMEOUT,
                        message=f"Annotation blocked: worker startup failed. {init_error}",
                        stream_id=chan_id,
                        camera_id=chan_id,
                        task_id=task_id,
                        details={
                            "algorithm": algorithm,
                            "timeout_sec": _WORKER_INIT_TIMEOUT_SEC,
                            "init_error": init_error,
                        },
                    )
                    _validation_failure_response(failure)

            log.info(
                "[detection] Step 9 passed: all %d task worker(s) ready for cam=%s "
                "(%.1fs)",
                len(chan_tasks), chan_id,
                time.monotonic() - task_init_start,
            )

            # ── Spawn FrameBus ─────────────────────────────────────────────
            bus_ready = self._manager.Event()
            self._bus_ready_events[chan_id] = bus_ready

            live_overlay = build_live_stream_overlay(chan_tasks)
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
                live_overlay,
                ),
                kwargs={"bus_ready_event": bus_ready, "cpu_affinity": _cpu_affinity_for(chan_id)},
                daemon=True,
            )
            self._bus_processes[chan_id] = bus
            self._stop_events[chan_id]   = stop_event
            self._task_queues_ref[chan_id] = task_queues

            bus_init_start = time.monotonic()
            bus.start()

            # ── Step 8: Wait for FrameBus to signal model ready ───────────
            signalled = bus_ready.wait(timeout=_BUS_INIT_TIMEOUT_SEC)
            if not signalled:
                log.error(
                    "[detection] FrameBus init timeout: cam=%s timeout=%.1fs "
                    "— killing spawned processes",
                    chan_id, _BUS_INIT_TIMEOUT_SEC,
                )
                stop_event.set()
                bus.join(timeout=3)
                if bus.is_alive():
                    bus.terminate()
                self._kill_channel_processes(chan_id, stop_event)

                st = dict(self._shared_state.get(chan_id, {}))
                init_error = st.get("error") or (
                    f"FrameBus did not signal ready within {_BUS_INIT_TIMEOUT_SEC:.0f}s"
                )
                failure = ValidationFailure(
                    stage=8,
                    stage_name="model_initialized",
                    error_code=BUS_INIT_TIMEOUT,
                    message=f"Annotation blocked: model failed to initialize. {init_error}",
                    stream_id=chan_id,
                    camera_id=chan_id,
                    details={
                        "timeout_sec": _BUS_INIT_TIMEOUT_SEC,
                        "init_error": init_error,
                    },
                )
                _validation_failure_response(failure)

            bus_init_elapsed = time.monotonic() - bus_init_start
            log.info(
                "[detection] Step 8 passed: FrameBus ready for cam=%s (%.1fs)",
                chan_id, bus_init_elapsed,
            )

            # Record init latency in shared_state
            try:
                row = dict(self._shared_state.get(chan_id, {}))
                row["init_latency_ms"] = round(bus_init_elapsed * 1000, 1)
                row["state_updated_at"] = time.time()
                self._shared_state[chan_id] = row
            except Exception:
                pass

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

        log.info(
            "[detection] All 10 validation+init steps passed. "
            "cameras=%s tasks=%s — annotation running.",
            started_cameras, started_tasks,
        )

        return {
            "status" : "started",
            "cameras": started_cameras,
            "tasks"  : started_tasks,
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _kill_channel_processes(self, chan_id: str, stop_event) -> None:
        """Signal stop and forcibly terminate all processes for a channel."""
        try:
            stop_event.set()
        except Exception:
            pass
        for p in list(self._task_processes.get(chan_id, {}).values()):
            if p.is_alive():
                p.join(timeout=2)
                if p.is_alive():
                    p.terminate()

    # ── Zombie reaper ─────────────────────────────────────────────────────────

    def _reap_finished_processes(self) -> None:
        """
        Join (reap) any FrameBus or task-worker processes that have already
        exited so they do not linger as OS zombies.  Called opportunistically
        before start/stop operations.
        """
        for proc in list(self._bus_processes.values()):
            if not proc.is_alive() and proc.exitcode is not None:
                proc.join(timeout=0)
        for task_procs in list(self._task_processes.values()):
            for proc in list(task_procs.values()):
                if not proc.is_alive() and proc.exitcode is not None:
                    proc.join(timeout=0)

    # ── Stop ──────────────────────────────────────────────────────────────────

    def _stop(self, camera_id: Optional[str] = None):
        self._reap_finished_processes()
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
            log.info("[detection] Camera stopped: cam=%s", cam_id)

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
def _cpu_affinity_for(camera_id: str):
    """
    S-1: Return CPU core list for this camera, or None when affinity is off.
    CAMERA_CPU_AFFINITY=cam1:0,1;cam2:2,3 — semicolon-separated, colon-delimited
    camera_id:core,core assignments.  Unspecified cameras are not pinned.
    CPU_AFFINITY_ENABLED=true must also be set.
    """
    import os as _os
    if _os.getenv("CPU_AFFINITY_ENABLED", "false").lower() not in ("true", "1", "yes"):
        return None
    raw = _os.getenv("CAMERA_CPU_AFFINITY", "").strip()
    if not raw:
        return None
    for entry in raw.split(";"):
        entry = entry.strip()
        if ":" not in entry:
            continue
        cam, cores_str = entry.split(":", 1)
        if cam.strip() == camera_id:
            try:
                return [int(c.strip()) for c in cores_str.split(",") if c.strip()]
            except ValueError:
                return None
    return None


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
    live_overlay=None,
    bus_ready_event=None,
    cpu_affinity=None,
):
    # S-1: pin this process to specific CPUs when CPU_AFFINITY_ENABLED=true.
    import os as _os
    if _os.getenv("CPU_AFFINITY_ENABLED", "false").lower() in ("true", "1", "yes") and cpu_affinity:
        try:
            import psutil as _psutil
            _psutil.Process().cpu_affinity(cpu_affinity)
        except Exception as _aff_exc:
            import logging as _log
            _log.getLogger(__name__).warning(
                "[%s] CPU affinity pin failed: %s", camera_id, _aff_exc
            )

    from frame_bus import FrameBus

    try:
        bus = FrameBus(
            camera_id,
            rtsp_url,
            shared_state,
            stop_event,
            task_queues,
            embedding_queue,
            frame_seq_counter=frame_seq,
            frame_seq_lock=frame_seq_lock,
            bus_fatal_event=bus_fatal_event,
            live_overlay=live_overlay,
            bus_ready_event=bus_ready_event,
        )
    except Exception as exc:
        import logging as _log
        _log.getLogger(__name__).error(
            "[%s] FrameBus init failed (model/GPU): %s", camera_id, exc
        )
        try:
            shared_state[camera_id] = {
                "camera_id": camera_id,
                "running": False,
                "error": str(exc),
                "error_code": MODEL_INIT_FAILED,
                "stopped_reason": "model_init_failed",
                "state_updated_at": __import__("time").time(),
            }
        except Exception:
            pass
        if bus_fatal_event is not None:
            try:
                bus_fatal_event.set()
            except Exception:
                pass
        return

    bus.run()


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
