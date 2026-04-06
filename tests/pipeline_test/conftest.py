"""
Shared fixtures — reset singleton state so tests stay isolated.

Stub ``ultralytics`` before any service imports so API tests run without
the full YOLO/torch stack; production still uses real ultralytics.
"""

import sys
import types

if "ultralytics" not in sys.modules:
    _ultra = types.ModuleType("ultralytics")

    class _FakeYOLO:
        def __init__(self, *a, **k):
            self.names = {0: "person"}

        def predict(self, *a, **k):
            return []

    _ultra.YOLO = _FakeYOLO
    sys.modules["ultralytics"] = _ultra

import pytest
from fastapi.testclient import TestClient

from app import app


def _stop_detection_processes(detection):
    for cam_id, proc in list(getattr(detection, "_processes", {}).items()):
        if proc is not None and proc.is_alive():
            ev = getattr(detection, "_stop_events", {}).get(cam_id)
            if ev is not None:
                ev.set()
            proc.join(timeout=3)
    procs = getattr(detection, "_processes", None)
    if isinstance(procs, dict):
        procs.clear()
    evs = getattr(detection, "_stop_events", None)
    if isinstance(evs, dict):
        evs.clear()

    for cam_id, proc in list(getattr(detection, "_bus_processes", {}).items()):
        if proc is not None and proc.is_alive():
            ev = getattr(detection, "_stop_events", {}).get(cam_id)
            if ev is not None:
                ev.set()
            proc.join(timeout=3)
            for p in getattr(detection, "_task_processes", {}).get(cam_id, {}).values():
                if p is not None and p.is_alive():
                    p.join(timeout=3)
    buses = getattr(detection, "_bus_processes", None)
    if isinstance(buses, dict):
        buses.clear()
    tasks = getattr(detection, "_task_processes", None)
    if isinstance(tasks, dict):
        tasks.clear()
    evs2 = getattr(detection, "_stop_events", None)
    if isinstance(evs2, dict):
        evs2.clear()


@pytest.fixture
def client():
    """TestClient as context manager."""
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def reset_singletons():
    """Clear camera registry and detection state before each test."""
    from apis.cameras import camera_registry
    from apis.detection import detection

    camera_registry._cameras.clear()
    if hasattr(detection, "_pipeline_names"):
        detection._pipeline_names = []
    _stop_detection_processes(detection)
    try:
        detection._shared_state.clear()
    except Exception:
        pass

    yield

    camera_registry._cameras.clear()
    if hasattr(detection, "_pipeline_names"):
        detection._pipeline_names = []
    _stop_detection_processes(detection)
    try:
        detection._shared_state.clear()
    except Exception:
        pass
