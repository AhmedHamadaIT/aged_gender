"""MASK_HAIRNET_CHEF_HAT with mocked PPEService."""

from __future__ import annotations

import base64
import json
from types import SimpleNamespace

import cv2
import numpy as np

from services.detector import Detection


def _jpeg_b64(frame: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", frame)
    assert ok
    return base64.b64encode(buf.tobytes()).decode("ascii")


def test_ppe_violation_when_classes_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPTURE_DIR", str(tmp_path / "cap"))
    monkeypatch.setenv("SCENE_DIR", str(tmp_path / "scene"))
    monkeypatch.setenv("EVENTS_DIR", str(tmp_path / "evt"))

    import services.ppe as ppe_mod

    def _fake_init(self):
        pass

    def _fake_call(self, ctx):
        ctx["data"].setdefault("use_case", {})
        ctx["data"]["use_case"]["ppe"] = [SimpleNamespace(items=[])]
        return ctx

    monkeypatch.setattr(ppe_mod.PPEService, "__init__", _fake_init)
    monkeypatch.setattr(ppe_mod.PPEService, "__call__", _fake_call)

    from services.mask_hairnet_chef_hat import MaskHairnetChefHatTask

    zones = [
        {
            "zone_id": "kitchen",
            "point": [
                {"x": 0, "y": 0},
                {"x": 640, "y": 0},
                {"x": 640, "y": 480},
                {"x": 0, "y": 480},
            ],
        }
    ]
    cfg = {
        "taskId": 77,
        "taskName": "ppe_u",
        "algorithmType": "MASK_HAIRNET_CHEF_HAT",
        "channelId": "1",
        "enable": True,
        "threshold": 50,
        "areaPosition": json.dumps(zones),
        "detailConfig": {"alarmType": ["no_mask"]},
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
    task = MaskHairnetChefHatTask(cfg)

    import cv2

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    det = Detection(100, 100, 200, 240, 0, "person", 0.95, track_id=9)
    payload = {
        "detection": {"items": [det], "count": 1},
        "timestamp": "ts",
        "camera_id": "cam1",
        "frame_b64": _jpeg_b64(frame),
    }
    events = task(payload)
    assert len(events) >= 1
    assert events[0]["alert"]["type"] == "no_mask"
