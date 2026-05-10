"""DetectionSSEBridge StreamFilters behaviour (pure)."""

from __future__ import annotations

from apis.detection_stream import StreamFilters


def test_stream_filters_task_name_via_lookup():
    ev = {"taskId": 3, "eventType": "CROSS_LINE"}

    def lookup(tid: int):
        return {"taskName": "alpha"} if tid == 3 else None

    f = StreamFilters(task_name="alpha")
    assert f.matches(ev, task_lookup=lookup) is True

    f2 = StreamFilters(task_name="beta")
    assert f2.matches(ev, task_lookup=lookup) is False
