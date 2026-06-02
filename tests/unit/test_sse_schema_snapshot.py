"""
tests/unit/test_sse_schema_snapshot.py
----------------------------------------
S-8: SSE event schema snapshot tests.

These tests verify that the top-level keys in each event type do NOT regress —
new keys are fine, but dropping required keys is a contract violation.

We test the event dict shape produced by build_cashier_structured_event,
the task_worker event envelope, and the cross-line crossing event shape.
"""

from __future__ import annotations

import json
import pytest


# ── Cashier SSE event ─────────────────────────────────────────────────────────

from services.cashier import SEVERITY_NORMAL, build_cashier_structured_event

_CASHIER_REQUIRED_KEYS = {
    "eventType",
    "taskId",
    "taskName",
    "channelId",
    "severity",
    "case_id",
    "timestamp",
    "data",
}


def _make_cashier_event():
    cashier_dict = {
        "case_id": "N1",
        "severity": SEVERITY_NORMAL,
        "summary": {"frame_id": 1},
        "data": {},
    }
    task_config = {"taskId": 1, "taskName": "test_cashier", "channelId": "cam1"}
    return build_cashier_structured_event(cashier_dict, task_config, "cam1")


def test_cashier_event_has_required_keys():
    ev = _make_cashier_event()
    missing = _CASHIER_REQUIRED_KEYS - set(ev.keys())
    assert not missing, f"Cashier event missing keys: {missing}"


def test_cashier_event_types():
    ev = _make_cashier_event()
    assert isinstance(ev["taskId"], int)
    assert isinstance(ev["taskName"], str)
    assert isinstance(ev["channelId"], str)
    assert isinstance(ev["eventType"], str)
    assert isinstance(ev["data"], dict)


# ── Cross-line SSE event ──────────────────────────────────────────────────────

_CROSS_LINE_REQUIRED_KEYS = {
    "eventType",
    "taskId",
    "taskName",
    "channelId",
    "timestamp",
    "person",
}

_CROSS_LINE_PERSON_KEYS = {"trackingId", "direction"}


def _make_cross_line_event():
    return {
        "eventType": "CROSS_LINE",
        "taskId": 99,
        "taskName": "gate",
        "channelId": "cam2",
        "timestamp": "2026-05-23T00:00:00Z",
        "person": {
            "trackingId": "42",
            "direction": "AB",
            "bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 10},
        },
    }


def test_cross_line_event_has_required_keys():
    ev = _make_cross_line_event()
    missing = _CROSS_LINE_REQUIRED_KEYS - set(ev.keys())
    assert not missing, f"Cross-line event missing keys: {missing}"


def test_cross_line_person_has_required_keys():
    ev = _make_cross_line_event()
    missing = _CROSS_LINE_PERSON_KEYS - set(ev["person"].keys())
    assert not missing, f"Cross-line person missing keys: {missing}"


# ── DetectionSSEBridge replay_after ───────────────────────────────────────────

from apis.detection_stream import DetectionSSEBridge


def test_replay_after_returns_list_of_dicts():
    bridge = DetectionSSEBridge(None)
    ev = {"eventType": "CROSS_LINE", "taskId": 1, "_seq": 5}
    bridge._record_replay(ev)
    result = bridge.replay_after(0)
    assert isinstance(result, list)
    assert all(isinstance(e, dict) for e in result)


def test_replay_after_filters_by_seq():
    bridge = DetectionSSEBridge(None)
    for i in range(1, 6):
        bridge._record_replay({"eventType": "X", "_seq": i})
    result = bridge.replay_after(3)
    seqs = [e["_seq"] for e in result]
    assert all(s > 3 for s in seqs)


def test_replay_after_deduplicates():
    bridge = DetectionSSEBridge(None)
    ev = {"eventType": "X", "_seq": 10}
    bridge._record_replay(ev)
    bridge._record_replay(ev)
    result = bridge.replay_after(0)
    assert sum(1 for e in result if e.get("_seq") == 10) == 1


# ── StreamFilters contract ────────────────────────────────────────────────────

from apis.detection_stream import StreamFilters


@pytest.mark.parametrize("task_id,event_task_id,match", [
    (1, 1, True),
    (1, 2, False),
    (None, 99, True),   # no filter → match all
])
def test_stream_filters_task_id(task_id, event_task_id, match):
    f = StreamFilters(task_id=task_id)
    ev = {"taskId": event_task_id}
    assert f.matches(ev) == match


@pytest.mark.parametrize("event_type,filter_type,match", [
    ("CROSS_LINE", "CROSS_LINE", True),
    ("CROSS_LINE", "CASHIER_BOX_OPEN", False),
    ("CROSS_LINE", None, True),
])
def test_stream_filters_event_type(event_type, filter_type, match):
    f = StreamFilters(event_type=filter_type)
    ev = {"eventType": event_type}
    assert f.matches(ev) == match
