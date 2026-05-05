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

try:
    import redis as _redis_lib
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False


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


def run_task_worker(
    camera_id   : str,
    task_config : dict,
    task_queue,
    result_queue,
    stop_event,
):
    """
    Entry point for each task worker process.

    Args:
        camera_id    : Camera identifier (string version of channelId).
        task_config  : Full task config dict from TaskRegistry.
        task_queue   : Input queue — receives payload dicts from FrameBus.
        result_queue : Output queue — events are pushed here for SSE streaming.
        stop_event   : Shared event; set when this camera should stop.
    """
    from services import TASK_REGISTRY

    algorithm = task_config["algorithmType"]
    task_id   = task_config["taskId"]

    task        = TASK_REGISTRY[algorithm](task_config)
    redis_client = _make_redis_client()
    redis_channel = f"live:event:{camera_id}"

    print(f"[{camera_id}/{algorithm}/{task_id}] Worker started.")

    while not stop_event.is_set():
        try:
            payload = task_queue.get(timeout=1.0)
        except _queue.Empty:
            continue
        except Exception:
            continue

        try:
            events = task(payload) or []
        except Exception as e:
            print(f"[{camera_id}/{algorithm}/{task_id}] Error: {e}")
            continue

        for event in events:
            # Exactly one outbound path so SSE subscribers are not doubled:
            # DetectionSSEBridge reads BOTH result_queue and Redis when REDIS_URL is set.
            if redis_client is not None:
                try:
                    redis_client.publish(redis_channel, json.dumps(event))
                except Exception:
                    pass  # never block on Redis errors
            else:
                try:
                    result_queue.put_nowait(event)
                except Exception:
                    pass  # drop if consumer is too slow — never block task processing

    print(f"[{camera_id}/{algorithm}/{task_id}] Worker stopped.")
