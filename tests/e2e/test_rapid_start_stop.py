"""Start/stop churn (guards against zombie workers)."""

from __future__ import annotations

import pytest

from tests.integration.helpers import cross_line_task, register_camera, register_task, unique_cam


@pytest.mark.timeout(600)
def test_rapid_start_stop_three_cycles(integration_http):
    cam = unique_cam("churn")
    register_camera(integration_http, cam)
    register_task(integration_http, cross_line_task(93001, cam, "churn_task"))

    for _ in range(3):
        r = integration_http.post("/detection/start", params={"camera_id": cam})
        assert r.status_code == 200, r.text
        integration_http.post("/detection/stop", params={"camera_id": cam})
