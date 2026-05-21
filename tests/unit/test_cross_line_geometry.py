"""Synthetic centroid sequences for CrossLineTask line-crossing logic."""

from __future__ import annotations

import base64
import json

import cv2
import numpy as np

from services.cross_line import (
    CrossLineTask,
    line_segment_pixels,
    parse_effective_cross_lines,
)
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


def test_cross_line_survives_missed_detection_frames(tmp_path, monkeypatch):
    """Side state is retained across frames with no bbox so crossing still fires."""
    monkeypatch.setenv("CAPTURE_DIR", str(tmp_path / "cap"))
    monkeypatch.setenv("SCENE_DIR", str(tmp_path / "scene"))
    monkeypatch.setenv("EVENTS_DIR", str(tmp_path / "evt"))
    monkeypatch.setenv("CROSSLINE_SIDE_STATE_TTL_FRAMES", "10")

    cfg = {
        "taskId": 11,
        "taskName": "cross_occl",
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
    base = {
        "timestamp": "t",
        "camera_id": "cam1",
        "frame_b64": _jpeg_b64(frame),
    }
    det_above = Detection(40, 60, 60, 80, 0, "person", 0.9, track_id=7)
    det_below = Detection(40, 110, 60, 130, 0, "person", 0.9, track_id=7)

    assert task({**base, "frame_id": 1, "detection": {"items": [det_above], "count": 1}}) == []
    assert task({**base, "frame_id": 2, "detection": {"items": [], "count": 0}}) == []
    events = task({**base, "frame_id": 3, "detection": {"items": [det_below], "count": 1}})
    assert len(events) == 1
    assert events[0]["person"]["trackingId"] == "7"


def test_parse_normalized_line_coords_not_truncated_to_zero() -> None:
    """Fractional y (e.g. 0.5) must scale to mid-frame, not int(0.5)==0."""
    raw = json.dumps(
        [
            {
                "line_id": "N1",
                "point": [{"x": 0, "y": 0.5}, {"x": 1, "y": 0.5}],
                "direction": 0,
            }
        ]
    )
    lines = parse_effective_cross_lines(raw)
    assert len(lines) == 1
    assert lines[0]["coords_space"] == "normalized"
    p0, p1 = line_segment_pixels(lines[0], 480, 360)
    assert p0 == (0, 180)
    assert p1 == (480, 180)


def test_cross_line_crossing_with_normalized_midline(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CAPTURE_DIR", str(tmp_path / "cap"))
    monkeypatch.setenv("SCENE_DIR", str(tmp_path / "scene"))
    monkeypatch.setenv("EVENTS_DIR", str(tmp_path / "evt"))

    area = json.dumps(
        [
            {
                "line_id": "N1",
                "point": [{"x": 0, "y": 0.5}, {"x": 1, "y": 0.5}],
                "direction": 0,
            }
        ]
    )
    cfg = {
        "taskId": 12,
        "taskName": "cross_norm",
        "algorithmType": "CROSS_LINE",
        "channelId": "cam1",
        "enable": True,
        "threshold": 50,
        "areaPosition": area,
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
    frame = np.zeros((360, 480, 3), dtype=np.uint8)
    det_lo = Detection(40, 60, 60, 80, 0, "person", 0.9, track_id=55)
    det_hi = Detection(40, 200, 60, 220, 0, "person", 0.9, track_id=55)
    base = {
        "timestamp": "t",
        "camera_id": "cam1",
        "frame_b64": _jpeg_b64(frame),
        "frame_id": 1,
    }
    assert task({**base, "detection": {"items": [det_lo], "count": 1}}) == []
    events = task({**base, "frame_id": 2, "detection": {"items": [det_hi], "count": 1}})
    assert len(events) == 1
