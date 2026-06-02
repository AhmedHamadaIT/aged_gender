"""
task_worker.py
--------------
Generic task worker — runs a single task in its own process.

Reads frame payloads from its dedicated queue (fed by FrameBus),
calls the task, and emits each resulting event on exactly one path:

  - Redis ``live:event:{camera_id}`` when ``REDIS_URL`` is reachable (SSE via
    ``DetectionSSEBridge._redis_loop`` and ``WS /cameras/.../events``).
  - Otherwise ``result_queue`` (multiprocessing.Queue) for the SSE bridge's
    ``_bridge_loop`` (no Redis / local dev).

Lifecycle (step 9 of the validation sequence):
  1. Instantiate the task algorithm from TASK_REGISTRY.
  2. Set ``worker_ready_event`` so the parent knows init succeeded.
  3. Enter the processing loop — polling ``task_queue`` and ``task_validity_map``.
  4. When ``stop_event`` is set or the task is invalidated, drain the buffer
     and exit cleanly.

Runtime invalidation:
  Every ``TASK_VALIDITY_POLL_SEC`` (default 2 s) the worker checks
  ``task_validity_map[task_id]``.  If the task was disabled or deleted the
  worker emits a structured PIPELINE_ERROR event on SSE/result_queue so
  clients are notified, then exits cleanly without crashing.

Structured errors:
  All error paths now emit ``PIPELINE_ERROR`` events through the same
  ``_emit_event`` path as business events so SSE clients always see them.
  ``print()`` to stderr is preserved only as a last-resort fallback when the
  emit path itself is broken.

Tasks are responsible for their own local persistence (JSONL, images).
"""

import json
import os
import queue as _queue
import sys
import time
from typing import Any, Optional

try:
    import redis as _redis_lib
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False

from resilience.circuit_breaker import CircuitBreaker
from resilience.event_buffer import EventBuffer
from resilience.sequencer import next_seq
from utils.error_codes import (
    WORKER_INIT_FAILED,
    TASK_REMOVED_RUNTIME,
    TASK_DISABLED_RUNTIME,
    WORKER_ERROR,
    RuntimeError_ as StructuredRuntimeError,
)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _make_redis_client():
    """Create a sync Redis client from REDIS_URL env var. Returns None on failure."""
    if not _REDIS_AVAILABLE:
        return None
    try:
        url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        client = _redis_lib.Redis.from_url(url, socket_connect_timeout=2)
        client.ping()
        return client
    except Exception as exc:
        print(f"[task_worker] Redis unavailable — event publishing disabled ({exc})", flush=True)
        return None


def _update_shared_resilience(
    shared_state: Any,
    camera_id: str,
    *,
    events_buffered: int,
    circuit_state: str,
    events_replayed: int = 0,
) -> None:
    if shared_state is None:
        return
    try:
        if camera_id not in shared_state:
            return
        row = dict(shared_state[camera_id])
        row["events_buffered"] = events_buffered
        row["task_redis_circuit_state"] = circuit_state
        if events_replayed:
            row["events_replayed"] = int(row.get("events_replayed", 0)) + int(
                events_replayed
            )
        shared_state[camera_id] = row
    except Exception:
        pass


def run_task_worker(
    camera_id   : str,
    task_config : dict,
    task_queue,
    result_queue,
    stop_event,
    shared_state=None,
    event_seq_counter=None,
    event_seq_lock=None,
    *,
    worker_ready_event=None,
    task_validity_map=None,
):
    """
    Entry point for each task worker process.

    Args:
        camera_id          : Camera identifier (string version of channelId).
        task_config        : Full task config dict from TaskRegistry.
        task_queue         : Input queue — receives payload dicts from FrameBus.
        result_queue       : Output queue — events are pushed here for SSE streaming.
        stop_event         : Shared event; set when this camera should stop.
        shared_state       : Optional manager dict for resilience metrics.
        event_seq_counter  : Optional per-camera event sequencing.
        event_seq_lock     : Optional lock for event sequencing.
        worker_ready_event : Set after successful task instantiation (step 9).
        task_validity_map  : Manager dict polled for runtime task invalidation.
    """
    from services import TASK_REGISTRY

    algorithm = task_config["algorithmType"]
    task_id   = str(task_config["taskId"])

    # ── Step 9: Instantiate task and signal ready ──────────────────────────
    try:
        task = TASK_REGISTRY[algorithm](task_config)
    except KeyError:
        _record_init_failure(
            shared_state, camera_id, task_id,
            f"Unknown algorithmType '{algorithm}'. "
            f"Supported: {sorted(TASK_REGISTRY.keys())}",
        )
        return
    except Exception as exc:
        _record_init_failure(
            shared_state, camera_id, task_id,
            f"Task init failed ({algorithm}): {exc}",
        )
        return

    if worker_ready_event is not None:
        try:
            worker_ready_event.set()
        except Exception:
            pass

    redis_client = _make_redis_client()
    use_redis = redis_client is not None
    redis_channel = f"live:event:{camera_id}"
    redis_breaker = CircuitBreaker(
        name=f"task_redis:{camera_id}:{task_id}",
        failure_threshold=max(1, _env_int("REDIS_CIRCUIT_FAILURES", 5)),
        reset_timeout_sec=max(1.0, float(os.getenv("REDIS_CIRCUIT_RESET_SEC", "30"))),
    )
    buffer_max = _env_int("EVENT_BUFFER_MAX", 1000)
    spill_path = os.getenv("EVENT_BUFFER_SPILL_DB", "").strip() or None
    event_buffer = EventBuffer(max_memory=buffer_max, db_path=spill_path)
    drain_interval = max(0.05, _env_int("EVENT_BUFFER_DRAIN_INTERVAL_MS", 200) / 1000.0)
    redis_retries = max(1, _env_int("REDIS_EVENT_RETRY", 3))
    retry_sleep = max(0.01, float(os.getenv("REDIS_EVENT_RETRY_SLEEP_SEC", "0.05")))
    validity_poll_sec = max(0.5, float(os.getenv("TASK_VALIDITY_POLL_SEC", "2.0")))
    last_drain = time.monotonic()
    last_metrics = time.monotonic()
    last_validity_check = time.monotonic()

    print(f"[{camera_id}/{algorithm}/{task_id}] Worker started.", flush=True)

    # ── Emit helpers ──────────────────────────────────────────────────────

    # S-7: Redis Streams shadow — optional XADD alongside Pub/Sub.
    _streams_enabled = os.getenv("REDIS_STREAMS_ENABLED", "false").lower() in ("true", "1", "yes")
    _stream_key = f"live:events:{camera_id}"
    _stream_maxlen = max(1, int(os.getenv("REDIS_STREAMS_MAXLEN", "500")))

    def _shadow_xadd(event: dict, line: str) -> None:
        """S-7: shadow-write to Redis Stream alongside Pub/Sub."""
        if not _streams_enabled or redis_client is None:
            return
        try:
            redis_client.xadd(
                _stream_key,
                {"data": line},
                maxlen=_stream_maxlen,
                approximate=True,
            )
        except Exception:
            pass  # Streams are a shadow; Pub/Sub is authoritative.

    def _emit_event(event: dict) -> None:
        nonlocal redis_client, last_drain
        seq = next_seq(event_seq_counter, event_seq_lock)
        if seq:
            event = {**event, "_seq": seq}

        if use_redis:
            line = json.dumps(event, default=str)
            published = False
            if redis_client is None:
                redis_client = _make_redis_client()
            if redis_client is not None and redis_breaker.allow_request():
                for attempt in range(redis_retries):
                    try:
                        redis_client.publish(redis_channel, line)
                        redis_breaker.record_success()
                        published = True
                        break
                    except Exception:
                        redis_breaker.record_failure()
                        time.sleep(retry_sleep)
                        redis_client = _make_redis_client()
            if published:
                _shadow_xadd(event, line)
            else:
                ok, reason = event_buffer.append(event)
                if not ok:
                    _log_event_drop(camera_id, task_id, reason, event)
        else:
            try:
                result_queue.put_nowait(event)
            except Exception:
                ok, reason = event_buffer.append(event)
                if not ok:
                    _log_event_drop(camera_id, task_id, reason, event)

        now = time.monotonic()
        if now - last_drain >= drain_interval:
            last_drain = now
            _drain_buffer()

    def _emit_error_event(
        error_code: str,
        message: str,
        *,
        stage: Optional[str] = None,
        details: Optional[dict] = None,
    ) -> None:
        """Emit a structured PIPELINE_ERROR event on the SSE/result-queue path."""
        err = StructuredRuntimeError(
            error_code=error_code,
            message=message,
            camera_id=camera_id,
            task_id=task_id,
            stage=stage,
            details=details or {},
        )
        try:
            _emit_event(err.to_event_dict())
        except Exception as emit_exc:
            print(
                json.dumps({
                    "pipeline_error": True,
                    "error_code": error_code,
                    "message": message,
                    "camera_id": camera_id,
                    "task_id": task_id,
                    "emit_exception": str(emit_exc),
                }, default=str),
                file=sys.stderr,
                flush=True,
            )

    def _drain_buffer() -> None:
        nonlocal redis_client
        batch = event_buffer.popleft_batch(64)
        if not batch:
            return
        if not use_redis:
            replayed = 0
            for i, ev in enumerate(batch):
                try:
                    result_queue.put_nowait(ev)
                    replayed += 1
                except Exception:
                    for ev2 in batch[i:]:
                        event_buffer.append(ev2)
                    break
            if replayed:
                _update_shared_resilience(
                    shared_state, camera_id,
                    events_buffered=len(event_buffer),
                    circuit_state="n/a",
                    events_replayed=replayed,
                )
            return
        if redis_client is None:
            redis_client = _make_redis_client()
        if redis_client is None or not redis_breaker.allow_request():
            for ev in batch:
                event_buffer.append(ev)
            return
        replayed = 0
        for ev in batch:
            try:
                line = json.dumps(ev, default=str)
                redis_client.publish(redis_channel, line)
                redis_breaker.record_success()
                replayed += 1
            except Exception:
                redis_breaker.record_failure()
                try:
                    event_buffer.append(ev)
                except Exception:
                    pass
                break
        if replayed:
            _update_shared_resilience(
                shared_state, camera_id,
                events_buffered=len(event_buffer),
                circuit_state=redis_breaker.state_label(),
                events_replayed=replayed,
            )

    # ── Processing loop ───────────────────────────────────────────────────

    while not stop_event.is_set():

        # ── Runtime task validity check ────────────────────────────────
        now_mono = time.monotonic()
        if (
            task_validity_map is not None
            and now_mono - last_validity_check >= validity_poll_sec
        ):
            last_validity_check = now_mono
            invalidation_reason = _check_task_validity(task_validity_map, task_id)
            if invalidation_reason is not None:
                code, msg = invalidation_reason
                print(
                    f"[{camera_id}/{algorithm}/{task_id}] Runtime invalidation: {msg}",
                    flush=True,
                )
                _emit_error_event(
                    code,
                    f"Annotation stopped: {msg}",
                    stage="runtime_validity",
                    details={"task_id": task_id, "algorithm": algorithm,
                             "camera_id": camera_id},
                )
                break

        try:
            _raw = task_queue.get(timeout=1.0)
            # M-2: resolve FrameRef → dict when SHM fan-out is active.
            from utils.task_payload import resolve_payload
            payload = resolve_payload(_raw)
            # M-4: apply per-task confThreshold to filter detections.
            _per_task_conf = float(
                (task_config.get("detailConfig") or {}).get("confThreshold", 0.0)
            )
            if _per_task_conf > 0.0:
                _det = payload.get("detection") or {}
                _items = _det.get("items") or []
                _filtered = [d for d in _items if float(d.get("conf", 1.0)) >= _per_task_conf]
                if len(_filtered) != len(_items):
                    payload = {**payload, "detection": {**_det, "items": _filtered, "count": len(_filtered)}}
        except _queue.Empty:
            if time.monotonic() - last_drain >= drain_interval:
                last_drain = time.monotonic()
                _drain_buffer()
            if time.monotonic() - last_metrics >= 1.0:
                last_metrics = time.monotonic()
                _update_shared_resilience(
                    shared_state, camera_id,
                    events_buffered=len(event_buffer),
                    circuit_state=redis_breaker.state_label(),
                )
            continue
        except Exception:
            continue

        try:
            events = task(payload) or []
        except Exception as exc:
            err_msg = (
                f"[{camera_id}/{algorithm}/{task_id}] Frame processing error: {exc}"
            )
            print(err_msg, file=sys.stderr, flush=True)
            _emit_error_event(
                WORKER_ERROR,
                f"Frame processing error in task {task_id} ({algorithm}): {exc}",
                stage="frame_processing",
                details={"frame_id": payload.get("frame_id"), "exception": str(exc)},
            )
            continue

        for event in events:
            _emit_event(event)

        if time.monotonic() - last_metrics >= 1.0:
            last_metrics = time.monotonic()
            _update_shared_resilience(
                shared_state, camera_id,
                events_buffered=len(event_buffer),
                circuit_state=redis_breaker.state_label(),
            )

    # Final drain
    _drain_buffer()
    event_buffer.close()
    print(f"[{camera_id}/{algorithm}/{task_id}] Worker stopped.", flush=True)


# ── Module-level helpers (outside run_task_worker to keep it picklable) ────────

def _check_task_validity(task_validity_map, task_id: str) -> Optional[tuple]:
    """
    Check whether the task is still valid.
    Returns ``(error_code, message)`` if invalid, ``None`` if valid.
    """
    try:
        entry = task_validity_map.get(task_id)
        if entry is None:
            return (
                TASK_REMOVED_RUNTIME,
                f"task {task_id} was removed from the registry during live stream.",
            )
        if not entry.get("exists", True):
            return (
                TASK_REMOVED_RUNTIME,
                f"task {task_id} was removed from the registry during live stream.",
            )
        if not entry.get("enabled", True):
            return (
                TASK_DISABLED_RUNTIME,
                f"task {task_id} was disabled (enable=false) during live stream.",
            )
    except Exception:
        pass
    return None


def _record_init_failure(
    shared_state: Any, camera_id: str, task_id: str, message: str
) -> None:
    """Write a structured init-failure entry to shared_state so the parent can report it."""
    print(
        json.dumps({
            "pipeline_error": True,
            "error_code": WORKER_INIT_FAILED,
            "camera_id": camera_id,
            "task_id": task_id,
            "message": message,
        }, default=str),
        file=sys.stderr,
        flush=True,
    )
    if shared_state is None:
        return
    try:
        row = dict(shared_state.get(camera_id, {}))
        row["worker_init_error"] = message
        row["error"] = message
        row["state_updated_at"] = time.time()
        shared_state[camera_id] = row
    except Exception:
        pass


def _log_event_drop(camera_id: str, task_id: str, reason: str, event: dict) -> None:
    print(
        json.dumps({
            "resilience": True,
            "event_drop": {
                "camera_id": camera_id,
                "task_id": task_id,
                "reason": reason,
                "event": event,
            },
        }, default=str),
        file=sys.stderr,
        flush=True,
    )
