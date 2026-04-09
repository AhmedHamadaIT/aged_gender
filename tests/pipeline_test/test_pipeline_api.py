"""
Full pipeline smoke test: ``POST /detection/setup`` with every service in
``services.REGISTRY``, then core REST routes (no ``POST /detection/start`` — avoids RTSP workers).

See ``README.md`` for how to run this file and matching ``curl`` examples.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from apis.cashier import _sse
from services import REGISTRY

EXPECTED_SERVICES = frozenset(
    {"detector", "age_gender", "ppe", "mood", "cashier"},
)


@pytest.fixture
def cashier_isolated(monkeypatch, tmp_path):
    cfg = tmp_path / "zones.yaml"
    cfg.write_text(
        "zones:\n"
        "  ROI_CASHIER:\n    shape: rectangle\n    points: [[0,0],[0.5,1]]\n"
        "  ROI_CUSTOMER:\n    shape: rectangle\n    points: [[0.5,0],[1,1]]\n"
        "thresholds: {}\n"
    )
    ev = tmp_path / "evidence"
    ev.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("CASHIER_CONFIG", str(cfg))
    import apis.cashier as cashier_api

    monkeypatch.setattr(cashier_api, "_evidence_dir", ev)
    return cfg, ev


def test_full_pipeline_and_api_endpoints(client: TestClient, cashier_isolated):
    """Full REGISTRY pipeline on setup; cameras, detection, cashier; OpenAPI SSE paths; SSE wire shape."""
    assert frozenset(REGISTRY.keys()) == EXPECTED_SERVICES

    pipeline = list(REGISTRY.keys())
    r = client.post("/detection/setup", json={"pipeline": pipeline})
    assert r.status_code == 200
    body = r.json()
    assert body == {"status": "configured", "pipeline": pipeline}

    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"service": "Vision Pipeline API", "version": "1.0.0"}

    spec = client.get("/openapi.json").json().get("paths", {})
    for p in (
        "/detection/stream",
        "/cashier/stream/{camera_id}",
        "/cashier/stream/{camera_id}/only",
    ):
        assert p in spec

    r = client.post(
        "/cameras",
        json={"cameras": [{"id": "cam1", "url": "rtsp://example/stream"}]},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "configured"
    assert r.json()["cameras"]["cam1"] == "rtsp://example/stream"

    r = client.get("/cameras")
    assert r.status_code == 200
    assert r.json()["count"] == 1

    r = client.get("/detection/status")
    assert r.status_code == 200
    assert r.json()["cameras"] == {}

    r = client.post("/detection/stop")
    assert r.status_code == 409

    r = client.get("/cashier/status")
    assert r.status_code == 200
    r = client.get("/cashier/events?limit=5")
    assert r.status_code == 200
    assert "total" in r.json() and "events" in r.json()
    r = client.delete("/cashier/events")
    assert r.status_code == 200
    r = client.get("/cashier/evidence")
    assert r.status_code == 200
    r = client.get("/cashier/zones")
    assert r.status_code == 200
    assert "zones" in r.json()
    r = client.post("/cashier/zones", json={"thresholds": {"drawer_open_max_seconds": 31}})
    assert r.status_code == 200
    assert r.json()["status"] == "updated"
    r = client.post("/cashier/zones/reset")
    assert r.status_code == 200
    assert r.json()["status"] == "reset_to_default"

    r = client.get("/cashier/media/cam1/drawer_count")
    assert r.status_code == 200
    assert r.json() == {"camera_id": "cam1", "drawer_open_count": 0}

    ev_dir = cashier_isolated[1]
    tiny = ev_dir / "p.jpg"
    tiny.write_bytes(b"\xff\xd8\xff\xd9")
    r = client.get(f"/cashier/evidence/{tiny.relative_to(ev_dir).as_posix()}")
    assert r.status_code == 200

    r = client.delete("/cameras/cam1")
    assert r.status_code == 200

    det = f"data: {json.dumps({'camera_id': 'cam1', 'data': {}})}\n\n"
    assert det.startswith("data:")
    assert "event: connected" in _sse("connected", {"camera_id": "cam1", "alert_only": False})
