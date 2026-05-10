"""
task_worker.py
--------------
Generic task worker — runs a single task in its own process.

Reads frame payloads from its dedicated queue (fed by FrameBus),
calls the task, and emits each resulting event on exactly one path:

  - Redis ``live:event:{camera_id}`` when ``REDIS_URL`` is reachable (SSE via
    ``DetectionSSEBridge._redis_loop`` and ``WS /cameras/.../events``).
  - Otherwise ``result_queue`` (multiprocessing.Queue) for the SSE bridge’s
    ``_bridge_loop`` (no Redis / local dev).

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
        print(f"[task_worker] Redis unavailable — event publishing disabled ({exc})")
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
):
    """
    Entry point for each task worker process.

    Args:
        camera_id    : Camera identifier (string version of channelId).
        task_config  : Full task config dict from TaskRegistry.
        task_queue   : Input queue — receives payload dicts from FrameBus.
        result_queue : Output queue — events are pushed here for SSE streaming.
        stop_event   : Shared event; set when this camera should stop.
        shared_state : Optional manager dict for resilience metrics.
        event_seq_counter / event_seq_lock : Optional per-camera event sequencing.
    """
    from services import TASK_REGISTRY

    algorithm = task_config["algorithmType"]
    task_id   = task_config["taskId"]

    task = TASK_REGISTRY[algorithm](task_config)
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
    last_drain = time.monotonic()
    last_metrics = time.monotonic()

    print(f"[{camera_id}/{algorithm}/{task_id}] Worker started.")

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
            if not published:
                ok, reason = event_buffer.append(event)
                if not ok:
                    payload = {
                        "level": "warning",
                        "component": "task_worker",
                        "camera_id": camera_id,
                        "task_id": task_id,
                        "reason": reason,
                        "event": event,
                    }
                    print(
                        json.dumps(
                            {"resilience": True, "event_drop": payload},
                            default=str,
                        ),
                        file=sys.stderr,
                    )
        else:
            try:
                result_queue.put_nowait(event)
            except Exception:
                ok, reason = event_buffer.append(event)
                if not ok:
                    print(
                        json.dumps(
                            {
                                "resilience": True,
                                "event_drop": {
                                    "camera_id": camera_id,
                                    "task_id": task_id,
                                    "reason": reason,
                                },
                            },
                            default=str,
                        ),
                        file=sys.stderr,
                    )

        now = time.monotonic()
        if now - last_drain >= drain_interval:
            last_drain = now
            _drain_buffer()

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
                    shared_state,
                    camera_id,
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
                shared_state,
                camera_id,
                events_buffered=len(event_buffer),
                circuit_state=redis_breaker.state_label(),
                events_replayed=replayed,
            )

    while not stop_event.is_set():
        try:
            payload = task_queue.get(timeout=1.0)
        except _queue.Empty:
            if time.monotonic() - last_drain >= drain_interval:
                last_drain = time.monotonic()
                _drain_buffer()
            if time.monotonic() - last_metrics >= 1.0:
                last_metrics = time.monotonic()
                _update_shared_resilience(
                    shared_state,
                    camera_id,
                    events_buffered=len(event_buffer),
                    circuit_state=redis_breaker.state_label(),
                )
            continue
        except Exception:
            continue

        try:
            events = task(payload) or []
        except Exception as e:
            print(f"[{camera_id}/{algorithm}/{task_id}] Error: {e}")
            continue

        for event in events:
            _emit_event(event)

        if time.monotonic() - last_metrics >= 1.0:
            last_metrics = time.monotonic()
            _update_shared_resilience(
                shared_state,
                camera_id,
                events_buffered=len(event_buffer),
                circuit_state=redis_breaker.state_label(),
            )

    # Final drain
    _drain_buffer()
    event_buffer.close()
    print(f"[{camera_id}/{algorithm}/{task_id}] Worker stopped.")
