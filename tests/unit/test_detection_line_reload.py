"""DetectionResource cross-line hot reload (overlay store + worker respawn)."""

from __future__ import annotations

import multiprocessing
from unittest.mock import MagicMock, patch

import pytest

from apis.detection import DetectionResource


@pytest.fixture
def detection_resource():
    if multiprocessing.current_process().name != "MainProcess":
        pytest.skip("DetectionResource requires MainProcess")
    return DetectionResource()


def test_set_channel_live_overlay_stores_geometry(detection_resource):
    tasks = [
        {
            "taskId": 1,
            "algorithmType": "CROSS_LINE",
            "enable": True,
            "areaPosition": (
                '[{"line_id":"1","line_name":"L","point":[{"x":0,"y":10},{"x":100,"y":10}],"direction":0}]'
            ),
        }
    ]
    detection_resource._set_channel_live_overlay("cam1", tasks)
    stored = detection_resource._live_overlay_store.get("cam1")
    assert stored is not None
    assert len(stored.get("cross_lines", [])) == 1


def test_reload_cross_line_when_camera_not_running(detection_resource):
    task = {
        "taskId": 5,
        "channelId": "cam9",
        "algorithmType": "CROSS_LINE",
        "enable": True,
        "areaPosition": "[]",
    }
    out = detection_resource.reload_cross_line_task(5, task)
    assert out["applied"] is False
    assert out["reason"] == "camera_not_running"


def test_reload_cross_line_refreshes_overlay_and_worker(detection_resource, monkeypatch):
    cam_id = "cam_reload"
    task_id = 42
    task_cfg = {
        "taskId": task_id,
        "taskName": "t",
        "channelId": cam_id,
        "algorithmType": "CROSS_LINE",
        "enable": True,
        "areaPosition": (
            '[{"line_id":"1","line_name":"L","point":[{"x":0,"y":10},{"x":100,"y":10}],"direction":0}]'
        ),
        "detailConfig": {},
        "validWeekday": ["MONDAY"],
        "validStartTime": 0,
        "validEndTime": 86400000,
        "threshold": 50,
    }

    bus_proc = MagicMock()
    bus_proc.is_alive.return_value = True
    detection_resource._bus_processes[cam_id] = bus_proc

    old_worker = MagicMock()
    old_worker.is_alive.return_value = True
    detection_resource._task_processes[cam_id] = {str(task_id): old_worker}

    q = detection_resource._manager.Queue(maxsize=4)
    detection_resource._task_queues_ref[cam_id] = {str(task_id): q}
    detection_resource._stop_events[cam_id] = detection_resource._manager.Event()
    detection_resource._ensure_channel_ipc(cam_id)

    def _fake_process(*args, **kwargs):
        proc = MagicMock()
        ready = kwargs.get("kwargs", {}).get("worker_ready_event")

        def _start():
            if ready is not None:
                ready.set()

        proc.start = MagicMock(side_effect=_start)
        return proc

    with patch("apis.detection.task_registry") as tr, patch(
        "task_worker.run_task_worker"
    ), patch("apis.detection.multiprocessing.Process", side_effect=_fake_process):
        tr.get_enabled.return_value = [task_cfg]
        out = detection_resource.reload_cross_line_task(task_id, task_cfg)

    assert out["applied"] is True
    assert out["overlay_refreshed"] is True
    assert detection_resource._live_overlay_store.get(cam_id) is not None
    old_worker.terminate.assert_called_once()
