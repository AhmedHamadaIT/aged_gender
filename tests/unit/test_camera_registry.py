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


def test_patch_camera_url():
    uid = str(uuid.uuid4())[:8]
    cam_id = f"c-{uid}"
    with TestClient(app) as client:
        r = client.post(
            "/cameras",
            json={"cameras": [{"id": cam_id, "url": "rtsp://127.0.0.1:8554/old"}]},
        )
        assert r.status_code == 200

        r_patch = client.patch(
            f"/cameras/{cam_id}",
            json={"url": "rtsp://127.0.0.1:8554/new"},
        )
        assert r_patch.status_code == 200
        body = r_patch.json()
        assert body["status"] == "updated"
        assert body["camera_id"] == cam_id
        assert body["url"] == "rtsp://127.0.0.1:8554/new"

        listed = client.get("/cameras").json()
        row = next(c for c in listed["cameras"] if c["id"] == cam_id)
        assert row["url"] == "rtsp://127.0.0.1:8554/new"


def test_patch_camera_not_found():
    with TestClient(app) as client:
        r = client.patch(
            "/cameras/does-not-exist",
            json={"url": "rtsp://127.0.0.1:8554/x"},
        )
        assert r.status_code == 404


def test_patch_camera_rtsp_url_alias():
    uid = str(uuid.uuid4())[:8]
    cam_id = f"c-{uid}"
    with TestClient(app) as client:
        client.post(
            "/cameras",
            json={"cameras": [{"id": cam_id, "url": "rtsp://127.0.0.1:8554/a"}]},
        )
        r = client.patch(
            f"/cameras/{cam_id}",
            json={"rtsp_url": "rtsp://127.0.0.1:8554/b"},
        )
        assert r.status_code == 200
        assert r.json()["url"] == "rtsp://127.0.0.1:8554/b"
