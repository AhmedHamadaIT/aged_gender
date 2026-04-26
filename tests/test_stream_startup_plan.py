"""Tests for stream startup plan: task validation, start guard, status enrichment."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

import app as app_mod

_VALID_LINE = [
    {
        "line_id": "1",
        "line_name": "test",
        "point": [{"x": 0, "y": 0}, {"x": 100, "y": 0}],
        "direction": 0,
    }
]


@pytest.fixture
def client() -> TestClient:
    with TestClient(app_mod.app, raise_server_exceptions=True) as c:
        yield c


@pytest.fixture
def clear_tasks():
    from apis.tasks import task_registry

    old = dict(task_registry._tasks)
    task_registry._tasks.clear()
    yield
    task_registry._tasks.clear()
    task_registry._tasks.update(old)


def test_cross_line_enabled_requires_non_empty_area(client: TestClient, clear_tasks) -> None:
    r = client.post(
        "/api/tasks",
        json={
            "taskId": 1,
            "taskName": "c1",
            "algorithmType": "CROSS_LINE",
            "channelId": "11",
            "enable": True,
            "areaPosition": "[]",
        },
    )
    assert r.status_code == 400
    assert "areaPosition" in r.json()["detail"].lower() or "cross_line" in r.json()["detail"].lower()


def test_post_tasks_returns_updated_on_upsert(client: TestClient, clear_tasks) -> None:
    body = {
        "taskId": 7,
        "taskName": "c7",
        "algorithmType": "CROSS_LINE",
        "channelId": "11",
        "enable": True,
        "areaPosition": json.dumps(_VALID_LINE),
    }
    r1 = client.post("/api/tasks", json=body)
    assert r1.status_code == 200
    assert r1.json()["status"] == "created"
    r2 = client.post("/api/tasks", json=body)
    assert r2.status_code == 200
    assert r2.json()["status"] == "updated"


def test_detection_start_rejects_multichannel_without_all_channels(
    client: TestClient, monkeypatch
) -> None:
    monkeypatch.setattr(
        app_mod.task_registry,
        "get_enabled",
        lambda: [
            {
                "taskId": 1,
                "taskName": "a",
                "algorithmType": "CROSS_LINE",
                "channelId": "11",
                "enable": True,
            },
            {
                "taskId": 2,
                "taskName": "b",
                "algorithmType": "CROSS_LINE",
                "channelId": "2",
                "enable": True,
            },
        ],
    )
    monkeypatch.setattr(
        app_mod.camera_registry,
        "all",
        lambda: {"11": "rtsp://a", "2": "rtsp://b"},
    )

    r = client.post("/detection/start")
    assert r.status_code == 400
    assert "Multiple" in r.json()["detail"] or "all_channels" in r.json()["detail"]


def test_detection_status_includes_reconciled_fields(
    client: TestClient, monkeypatch
) -> None:
    fake = {
        "camera_id": "cam1",
        "rtsp_url": "rtsp://x",
        "running": True,
        "frame_count": 1,
        "fps": 1.0,
        "last_detections": 0,
        "total_detections": 0,
        "state_updated_at": 1e9,
    }

    def _fake_get():
        from schemas import CameraStatus, DetectionStatus

        return DetectionStatus(
            cameras={
                "cam1": CameraStatus(
                    **app_mod.detection.enrich_shared_camera_row("cam1", dict(fake))
                )
            }
        )

    monkeypatch.setattr(app_mod.detection, "on_get", _fake_get)
    out = client.get("/detection/status").json()["cameras"]["cam1"]
    assert "framebus_process_alive" in out
    assert "last_state_update_age_sec" in out
