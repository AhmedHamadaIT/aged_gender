"""Start/stop detection without asserting ML outputs (synthetic RTSP)."""

from __future__ import annotations

import time

import pytest

from tests.integration.helpers import (
    cross_line_task,
    register_camera,
    register_task,
    unique_cam,
)


@pytest.mark.timeout(180)
def test_start_stop_single_camera(integration_http):
    cam = unique_cam("life")
    register_camera(integration_http, cam)
    register_task(integration_http, cross_line_task(91001, cam))

    r = integration_http.post("/detection/start", params={"camera_id": cam})
    assert r.status_code == 200, r.text

    deadline = time.monotonic() + 60.0
    running = False
    while time.monotonic() < deadline:
        st = integration_http.get("/detection/status").json()
        c = st.get("cameras", {}).get(cam)
        if c and c.get("running"):
            running = True
            break
        time.sleep(0.5)

    assert running, "camera did not reach running=True within timeout"

    r2 = integration_http.post("/detection/stop", params={"camera_id": cam})
    assert r2.status_code == 200, r2.text

