"""Synthetic centroid sequences for CrossLineTask line-crossing logic."""

from __future__ import annotations

import base64
import json

import cv2
import numpy as np

from services.cross_line import CrossLineTask
from services.detector import Detection


def _jpeg_b64(frame: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", frame)
    assert ok
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _line_area(line_id: str = "L1", direction: int = 0) -> str:
    data = [
        {
            "line_id": line_id,
            "line_name": "test",
            "point": [{"x": 0, "y": 100}, {"x": 200, "y": 100}],
            "direction": direction,
        }
    ]
    return json.dumps(data)


def test_cross_line_emits_once_per_crossing_bidirectional(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPTURE_DIR", str(tmp_path / "cap"))
    monkeypatch.setenv("SCENE_DIR", str(tmp_path / "scene"))
    monkeypatch.setenv("EVENTS_DIR", str(tmp_path / "evt"))

    cfg = {
        "taskId": 9,
        "taskName": "cross_unit",
        "algorithmType": "CROSS_LINE",
        "channelId": "cam1",
        "enable": True,
        "threshold": 50,
        "areaPosition": _line_area(direction=0),
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
    task = CrossLineTask(cfg)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    det_lo = Detection(40, 60, 60, 80, 0, "person", 0.9, track_id=42)
    det_hi = Detection(40, 110, 60, 130, 0, "person", 0.9, track_id=42)

    payload = {
        "detection": {"items": [det_lo], "count": 1},
        "timestamp": "t1",
        "camera_id": "cam1",
        "frame_b64": _jpeg_b64(frame),
    }
    assert task(payload) == []

    payload["detection"] = {"items": [det_hi], "count": 1}
    events = task(payload)
    assert len(events) == 1
    assert events[0]["eventType"] == "CROSS_LINE"
    assert events[0]["person"]["trackingId"] == "42"


def test_cross_line_skips_untracked_person(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPTURE_DIR", str(tmp_path / "cap"))
    monkeypatch.setenv("SCENE_DIR", str(tmp_path / "scene"))
    monkeypatch.setenv("EVENTS_DIR", str(tmp_path / "evt"))

    cfg = {
        "taskId": 10,
        "taskName": "cross_ut",
        "algorithmType": "CROSS_LINE",
        "channelId": "cam1",
        "enable": True,
        "threshold": 50,
        "areaPosition": _line_area(direction=0),
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
    task = CrossLineTask(cfg)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    det = Detection(40, 110, 60, 130, 0, "person", 0.9, track_id=-1)
    out = task(
        {
            "detection": {"items": [det], "count": 1},
            "timestamp": "t1",
            "camera_id": "cam1",
            "frame_b64": _jpeg_b64(frame),
        }
    )
    assert out == []
