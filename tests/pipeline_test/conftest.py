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


@pytest.fixture
def client():
    """TestClient must be a context manager so startup runs and pipeline is wired."""
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def reset_singletons():
    """Clear camera registry and detection config before each test."""
    from apis.cameras import camera_registry
    from apis.detection import detection

    camera_registry._cameras.clear()
    detection._pipeline_names = []
    # Ensure no stray worker processes from a previous failed run
    for cam_id, proc in list(getattr(detection, "_processes", {}).items()):
        if proc.is_alive():
            ev = detection._stop_events.get(cam_id)
            if ev is not None:
                ev.set()
            proc.join(timeout=3)
    detection._processes.clear()
    detection._stop_events.clear()
    try:
        detection._shared_state.clear()
    except Exception:
        pass

    yield

    camera_registry._cameras.clear()
    detection._pipeline_names = []
    for cam_id, proc in list(getattr(detection, "_processes", {}).items()):
        if proc.is_alive():
            ev = detection._stop_events.get(cam_id)
            if ev is not None:
                ev.set()
            proc.join(timeout=3)
    detection._processes.clear()
    detection._stop_events.clear()
    try:
        detection._shared_state.clear()
    except Exception:
        pass
