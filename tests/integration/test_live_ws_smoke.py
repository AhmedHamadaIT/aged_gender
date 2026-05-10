"""Binary JPEG frames on WebSocket when Redis + pipeline are up."""

from __future__ import annotations

import asyncio
import os
import time

import pytest
import websockets

from tests.integration.helpers import (
    cross_line_task,
    register_camera,
    register_task,
    unique_cam,
)


@pytest.mark.timeout(240)
@pytest.mark.asyncio
async def test_live_ws_receives_bytes(integration_http):
    cam = unique_cam("live")
    register_camera(integration_http, cam)
    register_task(integration_http, cross_line_task(91002, cam))

    r = integration_http.post("/detection/start", params={"camera_id": cam})
    assert r.status_code == 200, r.text

    base = os.getenv("TEST_WS_BASE", "ws://127.0.0.1:9000").rstrip("/")
    uri = f"{base}/cameras/{cam}/live"

    got = 0
    t0 = time.monotonic()
    try:
        async with websockets.connect(uri, max_size=None) as ws:
            while got < 2 and (time.monotonic() - t0) < 45.0:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                except asyncio.TimeoutError:
                    continue
                if isinstance(msg, (bytes, bytearray)) and len(msg) > 100:
                    got += 1
    finally:
        integration_http.post("/detection/stop", params={"camera_id": cam})

    assert got >= 1, "expected at least one JPEG frame over WS live stream"
