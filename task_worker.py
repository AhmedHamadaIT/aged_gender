"""
task_worker.py
--------------
Generic task worker — runs a single task in its own process.

Reads frame payloads from its dedicated queue (fed by FrameBus),
calls the task, and pushes any resulting events to the shared result queue
(which feeds the SSE stream).

Tasks are responsible for their own local persistence (JSONL, images).
The result queue is used to notify the backend in real-time via SSE.
"""

import queue as _queue


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

    task = TASK_REGISTRY[algorithm](task_config)
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
            try:
                result_queue.put_nowait(event)
            except Exception:
                pass   # drop if consumer is too slow — never block task processing

    print(f"[{camera_id}/{algorithm}/{task_id}] Worker stopped.")
