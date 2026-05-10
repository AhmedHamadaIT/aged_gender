"""Example payloads validate minimal structural constraints."""

from __future__ import annotations

from apis.detection_stream import StreamFilters


def test_stream_filters_task_and_channel():
    ev = {"taskId": 5, "channelId": "cam1", "eventType": "CROSS_LINE", "taskName": "t"}
    f = StreamFilters(task_id=5, channel_id="cam1")
    assert f.matches(ev, task_lookup=lambda _tid: None) is True

    f2 = StreamFilters(task_id=99)
    assert f2.matches(ev, task_lookup=lambda _tid: None) is False


def sample_cross_line_event():
    return {
        "eventId": "abc",
        "eventType": "CROSS_LINE",
        "timestamp": 1,
        "taskId": 1,
        "taskName": "x",
        "channelId": "1",
        "person": {"trackingId": "7"},
    }


def test_cross_line_event_has_tracking_string():
    ev = sample_cross_line_event()
    assert isinstance(ev["person"]["trackingId"], str)
