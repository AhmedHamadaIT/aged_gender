"""HTTP helpers for integration tests."""

from __future__ import annotations

import json
import os
import uuid
from typing import Any, Dict

import httpx


def unique_cam(prefix: str = "it") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


def rtsp_fixture_url(stream: str = "lobby_cross_line") -> str:
    """RTSP URL reachable from the API container (MediaMTX service name)."""
    return os.getenv(
        "INTEGRATION_RTSP_URL_TEMPLATE",
        "rtsp://mediamtx:8554/{stream}",
    ).format(stream=stream)


def register_camera(client: httpx.Client, cam_id: str, url: str | None = None) -> Dict[str, Any]:
    if url is None:
        url = rtsp_fixture_url()
    r = client.post("/cameras", json={"cameras": [{"id": cam_id, "url": url}]})
    r.raise_for_status()
    return r.json()


def cross_line_task(task_id: int, channel_id: str, name: str = "it_cross") -> Dict[str, Any]:
    area = [
        {
            "line_id": "1",
            "line_name": "mid",
            "point": [{"x": 0, "y": 240}, {"x": 640, "y": 240}],
            "direction": 0,
        }
    ]
    return {
        "taskId": task_id,
        "taskName": name,
        "algorithmType": "CROSS_LINE",
        "channelId": channel_id,
        "enable": True,
        "threshold": 30,
        "areaPosition": json.dumps(area),
        "detailConfig": {},
        "validWeekday": [
            "MONDAY",
            "TUESDAY",
            "WEDNESDAY",
            "THURSDAY",
            "FRIDAY",
            "SATURDAY",
            "SUNDAY",
        ],
        "validStartTime": 0,
        "validEndTime": 86400000,
    }


def register_task(client: httpx.Client, body: Dict[str, Any]) -> Dict[str, Any]:
    r = client.post("/api/tasks", json=body)
    r.raise_for_status()
    return r.json()
