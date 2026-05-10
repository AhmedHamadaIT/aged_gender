"""Reachability checks against the integration stack."""

from __future__ import annotations


def test_health_ok(integration_http):
    r = integration_http.get("/health")
    assert r.status_code == 200


def test_detection_status_shape(integration_http):
    r = integration_http.get("/detection/status")
    assert r.status_code == 200
    body = r.json()
    assert "cameras" in body
