"""Two cameras briefly concurrent."""

from __future__ import annotations

import time

import pytest

from tests.integration.helpers import cross_line_task, register_camera, register_task, unique_cam


@pytest.mark.timeout(400)
def test_two_cameras_start(integration_http):
    c1 = unique_cam("m1")
    c2 = unique_cam("m2")
    register_camera(integration_http, c1)
    register_camera(integration_http, c2)
    register_task(integration_http, cross_line_task(94001, c1, "t1"))
    register_task(integration_http, cross_line_task(94002, c2, "t2"))

    r1 = integration_http.post("/detection/start", params={"camera_id": c1})
    r2 = integration_http.post("/detection/start", params={"camera_id": c2})
    assert r1.status_code == 200, r1.text
    assert r2.status_code == 200, r2.text
    time.sleep(5.0)
    integration_http.post("/detection/stop", params={"camera_id": c1})
    integration_http.post("/detection/stop", params={"camera_id": c2})
