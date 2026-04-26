"""
Tests for the task-name live stream feature.

Coverage
--------
* TaskRegistry.get_by_name / require_by_name — unit tests (no HTTP stack).
* WS /tasks/{task_name}/live routing — verified by inspecting the FastAPI app
  with TestClient (header-level checks) and by patching live_frames_ws so we
  never need a real Redis or RTSP camera.

All tests pass without a camera, Redis, or a real RTSP stream.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.testclient import TestClient
from fastapi import HTTPException
from starlette.websockets import WebSocketDisconnect

from apis.tasks import TaskRegistry, TaskConfig


# ─────────────────────────────────────────────
# TaskRegistry.get_by_name / require_by_name
# ─────────────────────────────────────────────

def _make_registry(*configs: dict) -> TaskRegistry:
    """Build an isolated registry populated with the given task configs."""
    reg = TaskRegistry()
    for cfg in configs:
        reg.upsert(TaskConfig(**cfg))
    return reg


# Valid CROSS_LINE areaPosition for enabled tasks (see apis.tasks validation).
_AP_LINE = json.dumps(
    [
        {
            "line_id": "1",
            "line_name": "L",
            "point": [{"x": 0, "y": 0}, {"x": 100, "y": 0}],
            "direction": 0,
        }
    ]
)

_BASE = {
    "taskId": 1,
    "taskName": "entrance_line",
    "algorithmType": "CROSS_LINE",
    "channelId": "cam1",
    "areaPosition": _AP_LINE,
}


def test_task_config_channel_id_int_coerced_to_str():
    cfg = TaskConfig(**{**_BASE, "channelId": 1})
    assert cfg.channelId == "1"
    dumped = cfg.model_dump()
    assert dumped["channelId"] == "1"


def test_get_by_name_found():
    reg = _make_registry(_BASE)
    task = reg.get_by_name("entrance_line")
    assert task is not None
    assert task["channelId"] == "cam1"


def test_get_by_name_not_found():
    reg = _make_registry(_BASE)
    assert reg.get_by_name("nonexistent") is None


def test_get_by_name_ambiguous_raises_409():
    reg = _make_registry(
        {**_BASE, "taskId": 1, "taskName": "shared"},
        {**_BASE, "taskId": 2, "taskName": "shared"},
    )
    with pytest.raises(HTTPException) as exc_info:
        reg.get_by_name("shared")
    assert exc_info.value.status_code == 409
    assert "shared" in exc_info.value.detail
    assert "2 tasks" in exc_info.value.detail


def test_require_by_name_found():
    reg = _make_registry(_BASE)
    task = reg.require_by_name("entrance_line")
    assert task["taskId"] == 1


def test_require_by_name_not_found_raises_404():
    reg = _make_registry(_BASE)
    with pytest.raises(HTTPException) as exc_info:
        reg.require_by_name("ghost_task")
    assert exc_info.value.status_code == 404
    assert "ghost_task" in exc_info.value.detail


def test_require_by_name_propagates_409():
    reg = _make_registry(
        {**_BASE, "taskId": 1, "taskName": "dup"},
        {**_BASE, "taskId": 2, "taskName": "dup"},
    )
    with pytest.raises(HTTPException) as exc_info:
        reg.require_by_name("dup")
    assert exc_info.value.status_code == 409


def test_get_by_name_exact_match_only():
    """Partial / prefix matches must not return a result."""
    reg = _make_registry(_BASE)
    assert reg.get_by_name("entrance") is None
    assert reg.get_by_name("ENTRANCE_LINE") is None


# ─────────────────────────────────────────────
# WS /tasks/{task_name}/live — routing tests
# ─────────────────────────────────────────────


@pytest.fixture
def _register_task(monkeypatch):
    """
    Register a task into the live task_registry singleton before the test and
    remove it afterwards so tests do not bleed state into each other.
    """
    from apis import tasks as tasks_mod

    registered_ids: list[int] = []

    def register(**kw) -> dict:
        cfg = TaskConfig(**{**_BASE, **kw})
        task = tasks_mod.task_registry.upsert(cfg)
        registered_ids.append(cfg.taskId)
        return task

    yield register

    for tid in registered_ids:
        tasks_mod.task_registry._tasks.pop(tid, None)


def test_task_live_route_unknown_name(_register_task):
    """Connecting with an unknown taskName should close with code 4004."""
    import app as app_mod

    with TestClient(app_mod.app) as client:
        with client.websocket_connect("/tasks/does_not_exist/live") as ws:
            with pytest.raises(WebSocketDisconnect) as exc_info:
                ws.receive_bytes()
        assert exc_info.value.code == 4004


def test_task_live_route_delegates_to_camera_stream(_register_task, monkeypatch):
    """
    When the task exists and has a unique name, the handler should resolve the
    channelId and call live_frames_ws with that camera id.

    Patch ``app.live_frames_ws`` — the name already bound inside app.py — rather
    than the attribute on the apis.ws_live module, because app.py uses a name
    import (``from apis.ws_live import live_frames_ws``).
    """
    _register_task(taskId=50, taskName="gate_north", channelId="cam42")

    called_with: list[str] = []

    async def _fake_live_frames_ws(ws, camera_id: str):
        called_with.append(camera_id)
        await ws.accept()
        await ws.close()

    import app as app_mod

    monkeypatch.setattr(app_mod, "live_frames_ws", _fake_live_frames_ws)

    with TestClient(app_mod.app) as client:
        with client.websocket_connect("/tasks/gate_north/live") as ws:
            with pytest.raises(WebSocketDisconnect):
                ws.receive_bytes()

    assert called_with == ["cam42"]


def test_task_live_route_ambiguous_name_closes_4009(_register_task):
    """Ambiguous (duplicate) taskName must close with code 4009."""
    _register_task(taskId=60, taskName="shared_dup")
    _register_task(taskId=61, taskName="shared_dup")

    import app as app_mod

    with TestClient(app_mod.app) as client:
        with client.websocket_connect("/tasks/shared_dup/live") as ws:
            with pytest.raises(WebSocketDisconnect) as exc_info:
                ws.receive_bytes()
    assert exc_info.value.code == 4009


def test_task_live_route_distinct_names_resolve_independently(_register_task, monkeypatch):
    """Two tasks with different names each resolve to their own channelId."""
    _register_task(taskId=70, taskName="entrance_a", channelId="cam10")
    _register_task(taskId=71, taskName="entrance_b", channelId="cam20")

    resolved: list[str] = []

    async def _fake_live_frames_ws(ws, camera_id: str):
        resolved.append(camera_id)
        await ws.accept()
        await ws.close()

    import app as app_mod

    monkeypatch.setattr(app_mod, "live_frames_ws", _fake_live_frames_ws)

    with TestClient(app_mod.app) as client:
        for name in ("entrance_a", "entrance_b"):
            with client.websocket_connect(f"/tasks/{name}/live") as ws:
                with pytest.raises(WebSocketDisconnect):
                    ws.receive_bytes()

    assert "cam10" in resolved
    assert "cam20" in resolved
