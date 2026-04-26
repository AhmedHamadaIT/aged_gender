"""Tests for FrameBus task queue backpressure handling."""

from __future__ import annotations

import queue

from frame_bus import FrameBus


def _frame_bus_for_queue_tests(threshold: float = 0.8) -> FrameBus:
    bus = FrameBus.__new__(FrameBus)
    bus._task_queue_coalesce = True
    bus._task_queue_coalesce_threshold = threshold
    bus._task_queue_maxsize = 5
    bus._task_queue_coalesced_by_task = {"1": 0}
    bus._frames_dropped = 0
    bus._task_queue_drops_by_task = {"1": 0}
    bus._queue_warn_last = {}
    bus._queue_warn_interval = 9999.0
    bus.camera_id = "cam"
    return bus


def test_coalesce_starts_before_queue_is_full():
    bus = _frame_bus_for_queue_tests()
    q: queue.Queue = queue.Queue(maxsize=5)
    for item in range(4):
        q.put_nowait({"frame_id": item})

    bus._enqueue_task_payload(q, "1", {"frame_id": 99})

    assert q.qsize() == 4
    assert bus._task_queue_coalesced_by_task["1"] == 1
    assert bus._task_queue_drops_by_task["1"] == 0
    assert q.get_nowait()["frame_id"] == 1


def test_full_queue_coalesces_and_retries_latest_payload():
    bus = _frame_bus_for_queue_tests()
    q: queue.Queue = queue.Queue(maxsize=5)
    for item in range(5):
        q.put_nowait({"frame_id": item})

    bus._enqueue_task_payload(q, "1", {"frame_id": 99})

    assert q.qsize() == 5
    assert bus._task_queue_coalesced_by_task["1"] == 1
    assert bus._task_queue_drops_by_task["1"] == 0
    assert q.get_nowait()["frame_id"] == 1
