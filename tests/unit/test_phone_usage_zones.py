"""PHONE_USAGE with mocked PhoneService."""

from __future__ import annotations

import base64
from dataclasses import dataclass

import cv2
import numpy as np

from services.detector import Detection


def _jpeg_b64(frame: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", frame)
    assert ok
    return base64.b64encode(buf.tobytes()).decode("ascii")


@dataclass
class _PhoneHit:
    phone_detected: bool
    items: list


def test_phone_usage_inside_zone_emits(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPTURE_DIR", str(tmp_path / "cap"))
    monkeypatch.setenv("SCENE_DIR", str(tmp_path / "scene"))
    monkeypatch.setenv("EVENTS_DIR", str(tmp_path / "evt"))

    import services.phone as phone_mod

    def _fake_phone_init(self):
        pass

    def _fake_phone_call(self, ctx):
        ctx["data"].setdefault("use_case", {})
        ctx["data"]["use_case"]["phone"] = [
            _PhoneHit(
                True,
                [{"confidence": 0.95, "x1": 5, "y1": 5, "x2": 20, "y2": 25}],
            )
        ]
        return ctx

    monkeypatch.setattr(phone_mod.PhoneService, "__init__", _fake_phone_init)
    monkeypatch.setattr(phone_mod.PhoneService, "__call__", _fake_phone_call)

    from services.phone_usage import PhoneUsageTask

    zones = [
        {
            "zone_id": "z1",
            "point": [
                {"x": 0, "y": 0},
                {"x": 300, "y": 0},
                {"x": 300, "y": 300},
                {"x": 0, "y": 300},
            ],
        }
    ]
    import json

    cfg = {
        "taskId": 44,
        "taskName": "phone_u",
        "algorithmType": "PHONE_USAGE",
        "channelId": "1",
        "enable": True,
        "threshold": 50,
        "areaPosition": json.dumps(zones),
        "detailConfig": {},
        "validWeekday": [
            "MONDAY",
            "TUESDAY",
            "WEDNESDAY",
            "THURSDAY",
            "FRIDAY",
            "SATURDAY",
            "SUNDAY",
        ],
        "validStartTime": 0,
        "validEndTime": 86400000,
    }
    task = PhoneUsageTask(cfg)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    det = Detection(100, 100, 200, 240, 0, "person", 0.95, track_id=3)
    payload = {
        "detection": {"items": [det], "count": 1},
        "timestamp": "ts",
        "camera_id": "cam1",
        "frame_b64": _jpeg_b64(frame),
    }
    events = task(payload)
    assert len(events) == 1
    assert events[0]["eventType"] == "PHONE_USAGE"
    assert events[0]["person"]["trackingId"] == "3"


def test_phone_usage_outside_zone_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPTURE_DIR", str(tmp_path / "cap"))
    monkeypatch.setenv("SCENE_DIR", str(tmp_path / "scene"))
    monkeypatch.setenv("EVENTS_DIR", str(tmp_path / "evt"))

    import services.phone as phone_mod

    monkeypatch.setattr(phone_mod.PhoneService, "__init__", lambda self: None)

    def _no_phone(self, ctx):
        raise AssertionError("phone inference should not run outside zone")

    monkeypatch.setattr(phone_mod.PhoneService, "__call__", _no_phone)

    from services.phone_usage import PhoneUsageTask

    import json

    zones = [
        {
            "zone_id": "z1",
            "point": [
                {"x": 0, "y": 0},
                {"x": 50, "y": 0},
                {"x": 50, "y": 50},
                {"x": 0, "y": 50},
            ],
        }
    ]
    cfg = {
        "taskId": 45,
        "taskName": "phone_o",
        "algorithmType": "PHONE_USAGE",
        "channelId": "1",
        "enable": True,
        "threshold": 50,
        "areaPosition": json.dumps(zones),
        "detailConfig": {},
        "validWeekday": [
            "MONDAY",
            "TUESDAY",
            "WEDNESDAY",
            "THURSDAY",
            "FRIDAY",
            "SATURDAY",
            "SUNDAY",
        ],
        "validStartTime": 0,
        "validEndTime": 86400000,
    }
    task = PhoneUsageTask(cfg)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    det = Detection(400, 400, 420, 460, 0, "person", 0.95, track_id=3)
    payload = {
        "detection": {"items": [det], "count": 1},
        "timestamp": "ts",
        "camera_id": "cam1",
        "frame_b64": _jpeg_b64(frame),
    }
    assert task(payload) == []
