"""Structured cashier event builder (no ONNX cashier model)."""

from __future__ import annotations

from services.cashier import SEVERITY_NORMAL, build_cashier_structured_event


def test_build_cashier_structured_event_minimal():
    cashier_dict = {
        "case_id": "N1",
        "severity": SEVERITY_NORMAL,
        "summary": {"frame_id": 1},
        "data": {},
    }
    task_config = {"taskId": 12, "taskName": "cashier_a", "channelId": "cam-x"}
    ev = build_cashier_structured_event(cashier_dict, task_config, "cam-x")
    assert ev["eventType"] == "CASHIER_BOX_OPEN"
    assert ev["taskId"] == 12
    assert ev["channelId"] == "cam-x"
    assert "data" in ev
