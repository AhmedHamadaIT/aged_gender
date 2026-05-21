"""Eyego-shaped MASK_HAIRNET_CHEF_HAT structured events."""

from __future__ import annotations

import json

from services.mask_hairnet_chef_hat import (
    ALGORITHM_TYPE,
    build_ppe_person_structural,
    build_ppe_spec_data,
)


def test_build_ppe_person_structural_area_points_is_string():
    points = [{"x": 1004, "y": 56}, {"x": 1831, "y": 89}]
    raw = build_ppe_person_structural("no_chef_hat", points, (1417, 115, 1534, 259), 79)
    ps = json.loads(raw)
    assert ps["alarmType"] == "no_chef_hat"
    assert ps["score"] == 79
    assert ps["objectX"] == 1417
    assert ps["objectY"] == 115
    assert ps["objectWidth"] == 117
    assert ps["objectHeight"] == 144
    assert isinstance(ps["areaPoints"], str)
    assert json.loads(ps["areaPoints"]) == points


def test_build_ppe_spec_data_shape(monkeypatch):
    monkeypatch.setenv(
        "PPE_CLOUD_IMAGE_BASE",
        "https://storage.googleapis.com/logs-data-images",
    )
    data = build_ppe_spec_data(
        alarm_type="no_chef_hat",
        area_points=[{"x": 1004, "y": 56}],
        bbox=(1417, 115, 1534, 259),
        score=79,
        task_id=8,
        task_name="staff_safety_bar_area",
        channel_id=7,
        channel_name="7",
        device_sn="HQDZW1SBCABAH0235",
        record_ms=1774312985135,
    )
    assert data["algorithmType"] == ALGORITHM_TYPE
    assert data["taskId"] == 8
    assert data["taskName"] == "staff_safety_bar_area"
    assert data["channelId"] == 7
    assert data["channelName"] == "7"
    assert data["deviceSN"] == "HQDZW1SBCABAH0235"
    assert data["recordTime"] == 1774312985135
    assert data["dateUTC"] == "2026-03-24T00:43:05.135Z"
    assert data["captureId"].startswith(f"{ALGORITHM_TYPE}_")
    assert data["sceneId"].startswith(f"{ALGORITHM_TYPE}_")
    assert len(data["id"]) == 32
    assert data["captureUrl"].endswith(f"{data['id']}.jpg")
    assert data["captureUrl"].startswith(
        "https://storage.googleapis.com/logs-data-images/"
    )
    ps = json.loads(data["personStructural"])
    assert ps["alarmType"] == "no_chef_hat"
    assert "evidence" in data
    assert data["evidence"]["captureImage"]["type"] == "capture"
