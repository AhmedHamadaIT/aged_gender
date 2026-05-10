"""
Shared pytest fixtures for unit, contract, integration, and e2e tests.

Integration/e2e tests expect services from docker-compose.test.yml when
INTEGRATION_STACK=1 (individual tests use ``live_stack`` fixture).

Environment:
  INTEGRATION_STACK   If "1", wait for Redis/Qdrant/MediaMTX/API (required for integration/e2e).
  AUTO_COMPOSE        If "1", session-scoped fixture runs ``docker compose -f docker-compose.test.yml up -d --build``.
  AUTO_COMPOSE_DOWN   If "1", tears down compose after session (destructive).
  TEST_API_BASE       Base URL for HTTP client (default http://127.0.0.1:9000).
  REDIS_URL           Default redis://127.0.0.1:6379/0 for cleanup helpers.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional
from urllib.request import urlopen
from urllib.error import URLError

import pytest

# Repo root (parent of tests/)
ROOT = Path(__file__).resolve().parents[1]
COMPOSE_FILE = ROOT / "docker-compose.test.yml"


def pytest_collection_modifyitems(config: pytest.Config, items: List[pytest.Function]) -> None:
    """Apply default markers by folder so ``--strict-markers`` works."""
    for item in items:
        path = str(item.fspath)
        if "/tests/unit/" in path:
            item.add_marker(pytest.mark.unit)
        elif "/tests/contract/" in path:
            item.add_marker(pytest.mark.contract)
        elif "/tests/integration/" in path:
            item.add_marker(pytest.mark.integration)
        elif "/tests/e2e/" in path:
            item.add_marker(pytest.mark.e2e)
            node = item.nodeid.lower()
            if "soak" in node or "test_soak" in path.lower():
                item.add_marker(pytest.mark.soak)
        elif "/tests/test_" in path:
            # Legacy flat tests → unit
            item.add_marker(pytest.mark.unit)


def _wait_port(host: str, port: int, *, timeout_s: float = 60.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2.0):
                return
        except OSError:
            time.sleep(0.3)
    raise RuntimeError(f"Timeout waiting for {host}:{port}")


def _wait_http(url: str, *, timeout_s: float = 60.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urlopen(url, timeout=2.0) as r:
                if getattr(r, "status", 200) < 500:
                    return
        except URLError:
            time.sleep(0.3)
    raise RuntimeError(f"Timeout waiting for HTTP {url}")


@pytest.fixture(scope="session")
def live_stack() -> Generator[None, None, None]:
    """
    Ensures external stack is reachable when INTEGRATION_STACK=1.

    Optionally brings compose up when AUTO_COMPOSE=1.
    """
    if os.getenv("INTEGRATION_STACK", "").lower() not in ("1", "true", "yes"):
        pytest.skip(
            "Set INTEGRATION_STACK=1 and start docker-compose.test.yml (see docs/TEST_PLAYBOOK.md)"
        )

    if os.getenv("AUTO_COMPOSE", "").lower() in ("1", "true", "yes"):
        if not COMPOSE_FILE.is_file():
            pytest.skip(f"Missing {COMPOSE_FILE}")
        subprocess.run(
            ["docker", "compose", "-f", str(COMPOSE_FILE), "up", "-d", "--build"],
            cwd=str(ROOT),
            check=True,
        )

    _wait_port("127.0.0.1", 6379, timeout_s=90.0)
    _wait_http("http://127.0.0.1:6333/", timeout_s=90.0)
    _wait_port("127.0.0.1", 8554, timeout_s=90.0)
    _wait_http(os.getenv("TEST_API_BASE", "http://127.0.0.1:9000") + "/health", timeout_s=120.0)

    yield

    if os.getenv("AUTO_COMPOSE_DOWN", "").lower() in ("1", "true", "yes"):
        subprocess.run(
            ["docker", "compose", "-f", str(COMPOSE_FILE), "down", "-v"],
            cwd=str(ROOT),
            check=False,
        )


@pytest.fixture
def api_base() -> str:
    return os.getenv("TEST_API_BASE", "http://127.0.0.1:9000").rstrip("/")


@pytest.fixture
def app_client():
    """FastAPI TestClient bound to the real app (imports full stack)."""
    from fastapi.testclient import TestClient

    from app import app

    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


def register_camera(client, cam_id: str, url: str) -> dict:
    from apis.cameras import CameraSetupRequest

    body = {"cameras": [{"id": cam_id, "url": url}]}
    r = client.post("/cameras", json=body)
    assert r.status_code in (200, 201), r.text
    return r.json()


def register_task(client, task_payload: dict) -> dict:
    r = client.post("/api/tasks", json=task_payload)
    assert r.status_code in (200, 201), r.text
    return r.json()


@pytest.fixture
def wait_for():
    def _wait(predicate: Callable[[], bool], *, timeout_s: float = 30.0, interval_s: float = 0.25) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(interval_s)
        return False

    return _wait


@pytest.fixture
async def events_ws_messages():
    """Async helper factory: collect JSON events from WS /cameras/{id}/events."""

    async def _collect(camera_id: str, *, base_ws: str, max_messages: int = 50, timeout_s: float = 15.0):
        import websockets

        uri = base_ws.rstrip("/") + f"/cameras/{camera_id}/events"
        out: List[dict] = []
        try:
            async with websockets.connect(uri, max_size=None) as ws:
                t0 = time.monotonic()
                while len(out) < max_messages and (time.monotonic() - t0) < timeout_s:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
                    except asyncio.TimeoutError:
                        continue
                    if isinstance(raw, bytes):
                        raw = raw.decode("utf-8", errors="replace")
                    try:
                        out.append(json.loads(raw))
                    except json.JSONDecodeError:
                        continue
        except Exception:
            pass
        return out

    return _collect


def rtsp_url(name: str) -> str:
    """RTSP URL for a path published on local MediaMTX (fixture name without extension optional)."""
    host = os.getenv("MEDIAMTX_HOST", "127.0.0.1")
    port = os.getenv("MEDIAMTX_RTSP_PORT", "8554")
    path = name.replace(".mp4", "")
    return f"rtsp://{host}:{port}/{path}"


@pytest.fixture
def reset_redis_keys():
    """Best-effort flush of test-related Redis keys (optional for isolation)."""
    url = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")

    def _run():
        try:
            import redis as redis_sync

            r = redis_sync.Redis.from_url(url)
            # Do not FLUSHDB in shared dev Redis; only documented pattern is prefix delete in tests that need it.
            return
        except Exception:
            return

    return _run


@pytest.fixture
def integration_http(api_base: str):
    """httpx.Client against TEST_API_BASE for integration tests outside TestClient."""
    if os.getenv("INTEGRATION_STACK", "").lower() not in ("1", "true", "yes"):
        pytest.skip(
            "Set INTEGRATION_STACK=1 and start docker-compose.test.yml "
            "(see docs/TEST_PLAYBOOK.md)"
        )
    import httpx

    with httpx.Client(base_url=api_base, timeout=60.0) as client:
        yield client


# ── Bug capture (failure diagnostics) ─────────────────────────────────────────
_ARTIFACT_STATE: Dict[str, Any] = {"events": [], "metrics_snapshots": []}


@pytest.fixture
def artifact_sink():
    return _ARTIFACT_STATE


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo):
    outcome = yield
    rep = outcome.get_result()
    if rep.when == "call" and rep.failed:
        try:
            _capture_failure_artifacts(item, rep)
        except Exception:
            pass


def _capture_failure_artifacts(item: pytest.Item, rep: pytest.TestReport) -> None:
    base = ROOT / "artifacts" / item.nodeid.replace("/", "_").replace("::", "__")
    base.mkdir(parents=True, exist_ok=True)
    (base / "failure.txt").write_text(str(rep.longrepr), encoding="utf-8")
    # Docker logs (best-effort)
    log_cmd = [
        "docker",
        "compose",
        "-f",
        str(COMPOSE_FILE),
        "logs",
        "--no-color",
        "--tail",
        "200",
    ]
    if COMPOSE_FILE.is_file():
        try:
            p = subprocess.run(log_cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=60)
            (base / "docker_compose_logs.txt").write_text(p.stdout + "\n" + p.stderr, encoding="utf-8")
        except Exception as e:
            (base / "docker_compose_logs.txt").write_text(f"(failed to collect) {e}", encoding="utf-8")
