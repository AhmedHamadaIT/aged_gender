"""
tests/security/test_path_traversal.py
---------------------------------------
Security tests:
  1. Path traversal via /evidence/{file_path:path} — must be rejected.
  2. API_AUTH_TOKEN bearer enforcement on mutating routes.
  3. UPLOAD_MAX_BYTES cap enforcement on image-upload endpoints.
"""

from __future__ import annotations

import io
import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Fresh app client with no auth token (default/permissive)."""
    # Ensure auth is disabled so non-auth tests work without a token.
    monkeypatch.delenv("API_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("PROMETHEUS_ENABLED", raising=False)

    # Prevent actual file-system and model loads.
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("EVIDENCE_DIR", str(tmp_path / "evidence"))

    # Reload utils.auth so token reads the env at import time.
    import importlib, utils.auth
    importlib.reload(utils.auth)

    import importlib, app as app_module
    importlib.reload(app_module)
    from app import app
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def auth_client(tmp_path, monkeypatch):
    """App client with API_AUTH_TOKEN set to 'test-secret'."""
    monkeypatch.setenv("API_AUTH_TOKEN", "test-secret")
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("EVIDENCE_DIR", str(tmp_path / "evidence"))

    import importlib, utils.auth
    importlib.reload(utils.auth)

    import importlib, app as app_module
    importlib.reload(app_module)
    from app import app
    return TestClient(app, raise_server_exceptions=False)


# ── Path traversal ─────────────────────────────────────────────────────────────

_TRAVERSAL_PATHS = [
    "../etc/passwd",
    "../../etc/shadow",
    "%2e%2e%2fetc%2fpasswd",
    "cam1/../../../../etc/hosts",
    "..%2F..%2Fetc%2Fpasswd",
]


@pytest.mark.parametrize("bad_path", _TRAVERSAL_PATHS)
def test_evidence_path_traversal_rejected(client, bad_path):
    """GET /evidence/<traversal> must return 400 or 404, never serve /etc/* files."""
    resp = client.get(f"/evidence/{bad_path}")
    assert resp.status_code in (400, 404), (
        f"Expected 400 or 404 for traversal path '{bad_path}', got {resp.status_code}"
    )
    body = resp.text.lower()
    # Must not leak real file contents.
    assert "root:" not in body
    assert "daemon:" not in body


def test_evidence_valid_path_returns_404_when_missing(client, tmp_path):
    """A well-formed but nonexistent path yields 404 (not 500)."""
    resp = client.get("/evidence/cam1/scene_20260523.jpg")
    assert resp.status_code == 404


# ── Bearer auth enforcement ────────────────────────────────────────────────────

@pytest.mark.parametrize("method,path,body", [
    ("post", "/cameras", {"camera_id": "x", "rtsp_url": "rtsp://x"}),
    ("post", "/api/tasks", {"taskId": 1, "algorithmType": "CROSS_LINE", "channelId": "c", "taskName": "t", "enable": True}),
])
def test_mutating_route_401_without_token(auth_client, method, path, body):
    """When API_AUTH_TOKEN is set, mutating routes must return 401 without a token."""
    fn = getattr(auth_client, method)
    resp = fn(path, json=body)
    assert resp.status_code == 401, (
        f"{method.upper()} {path} expected 401, got {resp.status_code}"
    )


@pytest.mark.parametrize("method,path,body", [
    ("post", "/cameras", {"camera_id": "x", "rtsp_url": "rtsp://x"}),
])
def test_mutating_route_allowed_with_valid_token(auth_client, method, path, body):
    """A valid bearer token must be accepted (business logic may still fail; we check ≠ 401)."""
    fn = getattr(auth_client, method)
    resp = fn(path, json=body, headers={"Authorization": "Bearer test-secret"})
    assert resp.status_code != 401, f"Valid token rejected: {resp.status_code}"


def test_readonly_route_no_auth_required(auth_client):
    """GET routes must remain accessible without a token even when auth is enabled."""
    resp = auth_client.get("/cameras")
    assert resp.status_code != 401


# ── Upload size guard ──────────────────────────────────────────────────────────

def test_upload_size_cap_rejects_large_body(monkeypatch, tmp_path):
    """Content-Length exceeding UPLOAD_MAX_BYTES must yield 413."""
    monkeypatch.setenv("UPLOAD_MAX_BYTES", "100")   # tiny limit for test
    monkeypatch.delenv("API_AUTH_TOKEN", raising=False)

    import importlib, utils.auth
    importlib.reload(utils.auth)

    import importlib, app as app_module
    importlib.reload(app_module)
    from app import app

    client = TestClient(app, raise_server_exceptions=False)
    big_content = b"x" * 200  # 200 bytes > 100 limit
    resp = client.post(
        "/person_search/search",
        data={"top_k": "1"},
        files={"file": ("face.jpg", io.BytesIO(big_content), "image/jpeg")},
        headers={"Content-Length": str(len(big_content) + 500)},  # declared oversize
    )
    assert resp.status_code == 413


def test_upload_size_cap_allows_small_body(monkeypatch, tmp_path):
    """Content-Length within UPLOAD_MAX_BYTES must not trigger the 413 guard."""
    monkeypatch.setenv("UPLOAD_MAX_BYTES", str(10 * 1024 * 1024))
    monkeypatch.delenv("API_AUTH_TOKEN", raising=False)

    import importlib, utils.auth
    importlib.reload(utils.auth)

    import importlib, app as app_module
    importlib.reload(app_module)
    from app import app

    client = TestClient(app, raise_server_exceptions=False)
    small_content = b"x" * 50
    resp = client.post(
        "/person_search/search",
        data={"top_k": "1"},
        files={"file": ("face.jpg", io.BytesIO(small_content), "image/jpeg")},
    )
    # 413 must NOT be returned (may be 422/500 due to missing model, but not 413)
    assert resp.status_code != 413
