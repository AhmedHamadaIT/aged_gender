"""End-to-end camera registry flow: POST → GET → PATCH → link task → DELETE."""

from __future__ import annotations

import json
import uuid

from fastapi.testclient import TestClient

from app import app

_VALID_LINE = json.dumps(
    [
        {
            "line_id": "1",
            "line_name": "L1",
            "point": [{"x": 0, "y": 0}, {"x": 100, "y": 0}],
            "direction": 0,
        }
    ]
)


def test_camera_api_full_flow(monkeypatch):
    """Exercise all camera HTTP routes without mocks (snapshot capture disabled)."""
    monkeypatch.setattr(
        "apis.cameras.camera_registry._capture_snapshot",
        lambda _cid, _url: None,
    )
    uid = str(uuid.uuid4())[:8]
    cam_id = f"flow-{uid}"
    task_id = 90000 + int(uid, 16) % 10000

    with TestClient(app) as client:
        # POST register
        r_post = client.post(
            "/cameras",
            json={"cameras": [{"id": cam_id, "url": "rtsp://127.0.0.1:8554/v1"}]},
        )
        assert r_post.status_code == 200
        assert r_post.json()["status"] == "configured"
        assert cam_id in r_post.json()["cameras"]

        # GET list
        r_get = client.get("/cameras")
        assert r_get.status_code == 200
        row = next(c for c in r_get.json()["cameras"] if c["id"] == cam_id)
        assert row["url"] == "rtsp://127.0.0.1:8554/v1"

        # PATCH update URL
        new_url = "rtsp://127.0.0.1:8554/v2"
        r_patch = client.patch(f"/cameras/{cam_id}", json={"url": new_url})
        assert r_patch.status_code == 200
        assert r_patch.json() == {
            "status": "updated",
            "camera_id": cam_id,
            "url": new_url,
        }

        r_get2 = client.get("/cameras")
        row2 = next(c for c in r_get2.json()["cameras"] if c["id"] == cam_id)
        assert row2["url"] == new_url

        # Create task then link via POST /cameras/{id}/tasks
        r_task = client.post(
            "/api/tasks",
            json={
                "taskId": task_id,
                "taskName": f"task-{uid}",
                "algorithmType": "CROSS_LINE",
                "channelId": "wrong-channel",
                "areaPosition": _VALID_LINE,
            },
        )
        assert r_task.status_code == 200

        r_link = client.post(
            f"/cameras/{cam_id}/tasks",
            json={"taskId": task_id, "enable": True},
        )
        assert r_link.status_code == 200
        assert r_link.json()["status"] == "updated"
        assert r_link.json()["camera_id"] == cam_id
        assert r_link.json()["task"]["channelId"] == cam_id

        # DELETE
        r_del = client.delete(f"/cameras/{cam_id}")
        assert r_del.status_code == 200
        assert r_del.json()["status"] == "removed"
        assert cam_id not in r_del.json()["remaining"]

        r_patch_404 = client.patch(
            f"/cameras/{cam_id}",
            json={"url": "rtsp://127.0.0.1:8554/gone"},
        )
        assert r_patch_404.status_code == 404
