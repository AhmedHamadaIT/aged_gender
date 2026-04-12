"""Unit tests for CASHIER_BOX_OPEN structured events (no ONNX / FrameBus)."""

from __future__ import annotations

import json

from services.cashier import (
    CASE_LEVEL_INFO,
    CASE_LEVEL_WARNING,
    build_cashier_structured_event,
)


def test_build_cashier_structured_event_shape():
    cashier_dict = {
        "case_id": "N3",
        "severity": "NORMAL",
        "summary": {
            "frame_id": 7,
            "transaction": True,
            "evidence_path": "/tmp/evidence.jpg",
        },
        "data": {
            "algorithmType": "CASHIER_BOX_OPEN",
            "channelId": 2,
            "taskId": 101,
            "taskName": "cashier_drawer_monitor",
            "personStructural": json.dumps(
                {"case_matched": "N3", "case_level": CASE_LEVEL_INFO},
                separators=(",", ":"),
            ),
            "recordTime": 1,
            "dateUTC": "2026-01-01T00:00:00.000Z",
        },
    }
    task_config = {"taskId": 101, "taskName": "cashier_drawer_monitor", "channelId": 2}
    ev = build_cashier_structured_event(cashier_dict, task_config, "cam-1")

    assert ev["eventType"] == "CASHIER_BOX_OPEN"
    assert ev["taskId"] == 101
    assert ev["channelId"] == 2
    assert ev["case_id"] == "N3"
    assert ev["severity"] == "NORMAL"
    assert ev["transaction"] is True
    assert "data" in ev and isinstance(ev["data"], dict)
    assert ev["data"]["algorithmType"] == "CASHIER_BOX_OPEN"
    ps = json.loads(ev["data"]["personStructural"])
    assert ps["case_matched"] == "N3"
    assert ev["evidence"]["captureImage"] == "/tmp/evidence.jpg"


def test_severity_to_case_level_mapping():
    from services.cashier import (
        CASE_LEVEL_CRITICAL,
        SEVERITY_ALERT,
        SEVERITY_CRITICAL,
        SEVERITY_NORMAL,
        _severity_to_case_level,
    )

    assert _severity_to_case_level(SEVERITY_NORMAL) == CASE_LEVEL_INFO
    assert _severity_to_case_level(SEVERITY_ALERT) == CASE_LEVEL_WARNING
    assert _severity_to_case_level(SEVERITY_CRITICAL) == CASE_LEVEL_CRITICAL
