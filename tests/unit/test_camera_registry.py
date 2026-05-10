"""Camera registry HTTP behaviour."""

from __future__ import annotations

import uuid

from fastapi.testclient import TestClient

from app import app


def test_get_cameras_without_snapshot(monkeypatch):
    monkeypatch.setattr(
        "apis.cameras.camera_registry._capture_snapshot",
        lambda _cid, _url: None,
    )
    uid = str(uuid.uuid4())[:8]
    with TestClient(app) as client:
        r = client.post(
            "/cameras",
            json={"cameras": [{"id": f"c-{uid}", "url": "rtsp://127.0.0.1:8554/x"}]},
        )
        assert r.status_code == 200
        r2 = client.get("/cameras")
        assert r2.status_code == 200
        data = r2.json()
        ids = {c["id"] for c in data["cameras"]}
        assert f"c-{uid}" in ids
