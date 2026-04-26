"""
Smoke tests for HTTP/WebSocket routes defined in app.py.

These tests patch route delegates so we can validate endpoint wiring without
starting camera workers, Redis subscribers, or ML models.
"""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import app as app_mod


@pytest.fixture
def client() -> TestClient:
    with TestClient(app_mod.app, raise_server_exceptions=True) as c:
        yield c


def test_openapi_contains_expected_http_paths(client: TestClient):
    paths = set(client.get("/openapi.json").json()["paths"].keys())
    expected = {
        "/",
        "/health",
        "/status",
        "/cameras",
        "/cameras/{cam_id}",
        "/api/tasks",
        "/api/tasks/{task_id}",
        "/detection/start",
        "/detection/stop",
        "/detection/stop/all",
        "/detection/status",
        "/detection/stream",
        "/person_search/search",
        "/person_search/health",
        "/semantic_search/search",
        "/semantic_search/health",
        "/stream/metrics",
        "/stream/quality-events",
        "/stream/live/{camera_id}",
        "/cashier/status",
        "/cashier/events",
        "/cashier/evidence",
        "/cashier/evidence/{file_path}",
        "/cashier/zones",
        "/cashier/zones/reset",
        "/cashier/stream/{camera_id}",
        "/cashier/stream/{camera_id}/only",
        "/cashier/media/{camera_id}/latest/jpg",
        "/cashier/media/{camera_id}/latest/gif",
        "/cashier/media/{camera_id}/event/{event_id}/jpg",
        "/cashier/media/{camera_id}/event/{event_id}/gif",
        "/cashier/media/{camera_id}/drawer_count",
    }
    assert expected.issubset(paths)


def test_basic_health_endpoints(client: TestClient, monkeypatch):
    monkeypatch.setattr(app_mod.detection, "on_get", lambda: {"cameras": {}})

    assert client.get("/").status_code == 200
    assert client.get("/health").status_code == 200
    assert client.get("/status").json() == {"cameras": {}}
    assert client.get("/detection/status").json() == {"cameras": {}}


def test_camera_endpoints(client: TestClient, monkeypatch):
    monkeypatch.setattr(
        app_mod.camera_registry,
        "on_post",
        lambda req: {"status": "configured", "count": len(req.cameras)},
    )
    monkeypatch.setattr(
        app_mod.camera_registry,
        "on_get",
        lambda: {"count": 1, "cameras": [{"id": "cam1", "url": "rtsp://cam1"}]},
    )
    monkeypatch.setattr(
        app_mod.camera_registry,
        "on_delete",
        lambda cam_id: {"status": "removed", "camera_id": cam_id},
    )

    r1 = client.post("/cameras", json={"cameras": [{"id": "cam1", "url": "rtsp://cam1"}]})
    assert r1.status_code == 200
    assert r1.json()["status"] == "configured"

    r2 = client.get("/cameras")
    assert r2.status_code == 200
    assert r2.json()["count"] == 1

    r3 = client.delete("/cameras/cam1")
    assert r3.status_code == 200
    assert r3.json()["camera_id"] == "cam1"


def test_task_endpoints(client: TestClient, monkeypatch):
    task = {"taskId": 9, "taskName": "t", "algorithmType": "CROSS_LINE", "channelId": "cam1"}
    monkeypatch.setattr(app_mod.task_registry, "on_post", lambda cfg: {"status": "created", "task": task})
    monkeypatch.setattr(app_mod.task_registry, "on_get_all", lambda: {"count": 1, "tasks": [task]})
    monkeypatch.setattr(app_mod.task_registry, "on_get_one", lambda task_id: {**task, "taskId": task_id})
    monkeypatch.setattr(app_mod.task_registry, "on_put", lambda task_id, cfg: {"status": "updated", "task": {**task, "taskId": task_id}})
    monkeypatch.setattr(app_mod.task_registry, "on_delete", lambda task_id: {"status": "deleted", "taskId": task_id})

    payload = {
        "taskId": 9,
        "taskName": "t",
        "algorithmType": "CROSS_LINE",
        "channelId": "cam1",
    }
    assert client.post("/api/tasks", json=payload).status_code == 200
    assert client.get("/api/tasks").json()["count"] == 1
    assert client.get("/api/tasks/9").json()["taskId"] == 9
    assert client.put("/api/tasks/9", json=payload).json()["status"] == "updated"
    assert client.delete("/api/tasks/9").json()["status"] == "deleted"


def test_detection_control_endpoints(client: TestClient, monkeypatch):
    monkeypatch.setattr(app_mod.detection, "on_post", lambda req: {"status": req.action, "camera_id": req.camera_id})

    start = client.post("/detection/start", params={"camera_id": "cam1"})
    assert start.status_code == 200
    assert start.json() == {"status": "start", "camera_id": "cam1"}

    stop = client.post("/detection/stop", params={"camera_id": "cam1"})
    assert stop.status_code == 200
    assert stop.json() == {"status": "stop", "camera_id": "cam1"}

    stop_all = client.post("/detection/stop/all")
    assert stop_all.status_code == 200
    assert stop_all.json() == {"status": "stop_all", "camera_id": None}


def test_stream_metrics_endpoint(client: TestClient):
    r = client.get("/stream/metrics")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_stream_metrics_returns_populated_camera(client: TestClient):
    """Metrics list mirrors detection shared_state (plain dict values)."""
    fake = {
        "camera_id": "cam-x",
        "rtsp_url": "rtsp://example/stream",
        "running": True,
        "frame_count": 100,
        "fps": 12.3,
        "fps_actual": 12.3,
        "last_detections": 1,
        "total_detections": 50,
        "uptime_seconds": 360.0,
        "uptime_sec": 360.0,
        "error": None,
        "stream_quality": "640x360@10fps",
        "frames_dropped": 2,
        "drop_rate": 0.03,
        "decode_error_rate": 0.03,
        "task_queue_drops": 2,
        "task_queue_drop_rate": 0.0198,
        "reconnects": 2,
        "latency_estimate_ms": 95.0,
        "stream_read_failures": 3,
        "decode_failures": 3,
        "decoder": "cpu",
        "hw_decoder_requested": None,
        "hw_decoder_active": False,
        "profile": "balanced",
        "transport": "tcp",
        "embed_skip_rate": 0.0,
    }
    try:
        app_mod.detection._shared_state.clear()
        app_mod.detection._shared_state["cam-x"] = fake

        r = client.get("/stream/metrics")
        assert r.status_code == 200
        rows = r.json()
        assert len(rows) == 1
        row = rows[0]
        assert row["camera_id"] == "cam-x"
        assert row["drop_rate"] == 0.03
        assert row["decode_error_rate"] == 0.03
        assert row["task_queue_drop_rate"] == 0.0198
        assert row["reconnects"] == 2
        assert row["uptime_sec"] == 360.0
        assert row["fps_actual"] == 12.3
    finally:
        app_mod.detection._shared_state.clear()


def test_person_search_endpoints(client: TestClient, monkeypatch):
    async def _fake_search(file, top_k: int):
        assert top_k == 3
        return {"status": "success", "count": 1, "results": [{"id": "p1"}]}

    monkeypatch.setattr(app_mod.person_search_api, "search", _fake_search)
    monkeypatch.setattr(app_mod.person_search_api.person_search_service, "model", object())

    fileobj = io.BytesIO(b"\xff\xd8\xff fake")
    r = client.post(
        "/person_search/search",
        files={"file": ("person.jpg", fileobj, "image/jpeg")},
        data={"top_k": "3"},
    )
    assert r.status_code == 200
    assert r.json()["count"] == 1

    health = client.get("/person_search/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"


def test_semantic_search_endpoints(client: TestClient, monkeypatch):
    async def _fake_search(text_query, file, top_k: int):
        assert text_query == "blue shirt"
        assert file is None
        assert top_k == 5
        return {"status": "success", "count": 1, "results": [{"id": "img1"}]}

    monkeypatch.setattr(app_mod.semantic_search_api, "search", _fake_search)
    monkeypatch.setattr(app_mod.semantic_search_api.semantic_search_service, "_ready", True)

    r = client.post("/semantic_search/search", data={"text_query": "blue shirt", "top_k": "5"})
    assert r.status_code == 200
    assert r.json()["count"] == 1

    health = client.get("/semantic_search/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"


def test_camera_websocket_endpoints_delegate(client: TestClient, monkeypatch):
    calls: list[tuple[str, str]] = []

    async def _fake_frames(ws, camera_id: str):
        calls.append(("frames", camera_id))
        await ws.accept()
        await ws.close()

    async def _fake_events(ws, camera_id: str):
        calls.append(("events", camera_id))
        await ws.accept()
        await ws.close()

    monkeypatch.setattr(app_mod, "live_frames_ws", _fake_frames)
    monkeypatch.setattr(app_mod, "live_events_ws", _fake_events)

    with client.websocket_connect("/cameras/cam-x/live") as ws_live:
        with pytest.raises(WebSocketDisconnect):
            ws_live.receive_bytes()

    with client.websocket_connect("/cameras/cam-y/events") as ws_events:
        with pytest.raises(WebSocketDisconnect):
            ws_events.receive_text()

    assert ("frames", "cam-x") in calls
    assert ("events", "cam-y") in calls
