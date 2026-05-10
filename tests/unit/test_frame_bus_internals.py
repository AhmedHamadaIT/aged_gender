"""FrameBus helpers not covered elsewhere."""

from __future__ import annotations

import queue
import time

import pytest

from frame_bus import FrameBus


def test_enqueue_drops_when_full_without_coalesce():
    bus = FrameBus.__new__(FrameBus)
    bus._task_queue_coalesce = False
    bus._task_queue_maxsize = 2
    bus._task_queue_coalesce_threshold = 0.5
    bus._task_queue_coalesced_by_task = {"9": 0}
    bus._frames_dropped = 0
    bus._task_queue_drops_by_task = {"9": 0}
    bus._queue_warn_last = {}
    bus._queue_warn_interval = 9999.0
    bus.camera_id = "cam"

    q: queue.Queue = queue.Queue(maxsize=2)
    q.put_nowait({"a": 1})
    q.put_nowait({"a": 2})

    bus._enqueue_task_payload(q, "9", {"a": 3})
    assert bus._frames_dropped >= 1


def test_enrich_shared_camera_row_optional_age():
    import apis.detection as det_mod

    if det_mod.detection is None:
        pytest.skip("detection API only in MainProcess")

    now = time.time()
    row = det_mod.detection.enrich_shared_camera_row(
        "cam_z",
        {"running": True, "state_updated_at": now},
    )
    assert row["framebus_process_alive"] is None
    assert row["last_state_update_age_sec"] is not None
