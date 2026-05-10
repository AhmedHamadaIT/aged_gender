"""Critical routes exist in OpenAPI schema."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import app


@pytest.fixture
def openapi_schema():
    with TestClient(app) as client:
        r = client.get("/openapi.json")
        assert r.status_code == 200
        return r.json()


def test_openapi_contains_core_routes(openapi_schema):
    paths = openapi_schema.get("paths", {})
    for route in (
        "/health",
        "/cameras",
        "/api/tasks",
        "/detection/start",
        "/detection/status",
        "/detection/stream",
        "/stream/metrics",
        "/person_search/search",
        "/semantic_search/search",
    ):
        assert route in paths, f"missing {route}"


def test_openapi_has_ws_documentation_in_description(openapi_schema):
    desc = (openapi_schema.get("info") or {}).get("description") or ""
    assert "WebSocket" in desc or "websocket" in desc.lower()
