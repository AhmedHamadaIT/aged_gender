"""
Integration-style tests for all HTTP routes under /cashier (apis/cashier.py).

Uses a minimal FastAPI app with only the cashier router so importing does not
start detection multiprocessing.
"""

from __future__ import annotations

import asyncio
import json

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

import apis.cashier as cashier_api
from apis.cashier import push_result, router as cashier_router


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────


@pytest.fixture
def app_cashier() -> FastAPI:
    app = FastAPI()
    app.include_router(cashier_router, prefix="/cashier")
    return app


@pytest.fixture
def client(app_cashier: FastAPI) -> TestClient:
    return TestClient(app_cashier)


def test_cashier_router_registers_expected_paths(app_cashier: FastAPI):
    paths = {getattr(route, "path", "") for route in app_cashier.routes}
    expected = {
        "/cashier/status",
        "/cashier/events",
        "/cashier/evidence",
        "/cashier/evidence/{file_path:path}",
        "/cashier/zones",
        "/cashier/zones/reset",
        "/cashier/stream/{camera_id}",
        "/cashier/stream/{camera_id}/only",
        "/cashier/media/{camera_id}/latest/jpg",
        "/cashier/media/{camera_id}/latest/gif",
        "/cashier/media/{camera_id}/event/{event_id}/jpg",
        "/cashier/media/{camera_id}/event/{event_id}/gif",
        "/cashier/media/{camera_id}/drawer_count",
    }
    for p in expected:
        assert p in paths


@pytest.fixture(autouse=True)
def _reset_cashier_state() -> None:
    cashier_api._last_result.clear()
    cashier_api._event_log.clear()
    yield


@pytest.fixture
def config_yaml(tmp_path, monkeypatch):
    """Writable zone config; CASHIER_CONFIG points here."""
    path = tmp_path / "zones.yaml"
    path.write_text(
        yaml.dump(
            {
                "zones": {
                    "ROI_CASHIER": {
                        "shape": "rectangle",
                        "points": [[0.0, 0.0], [0.4, 1.0]],
                        "active": True,
                    },
                    "ROI_CUSTOMER": {
                        "shape": "rectangle",
                        "points": [[0.4, 0.0], [1.0, 1.0]],
                        "active": True,
                    },
                },
                "thresholds": {"drawer_open_max_seconds": 25},
            },
            default_flow_style=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CASHIER_CONFIG", str(path))
    return path


@pytest.fixture
def evidence_dir(tmp_path, monkeypatch):
    d = tmp_path / "evidence"
    d.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(cashier_api, "_evidence_dir", d)
    return d


# ─────────────────────────────────────────────
# Zones
# ─────────────────────────────────────────────


def test_get_zones_returns_yaml_config(client: TestClient, config_yaml):
    r = client.get("/cashier/zones")
    assert r.status_code == 200
    data = r.json()
    assert "zones" in data
    assert "ROI_CASHIER" in data["zones"]
    assert data["thresholds"]["drawer_open_max_seconds"] == 25


def test_post_zones_partial_merge(client: TestClient, config_yaml):
    r = client.post(
        "/cashier/zones",
        json={
            "ROI_CASHIER": {
                "shape": "rectangle",
                "points": [{"x": 0.0, "y": 0.0}, {"x": 0.35, "y": 1.0}],
                "active": True,
            },
            "thresholds": {"drawer_open_max_seconds": 18},
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "updated"
    assert body["config"]["zones"]["ROI_CASHIER"]["points"] == [[0.0, 0.0], [0.35, 1.0]]
    assert body["config"]["thresholds"]["drawer_open_max_seconds"] == 18
    # customer zone unchanged from initial file
    assert body["config"]["zones"]["ROI_CUSTOMER"]["points"] == [[0.4, 0.0], [1.0, 1.0]]

    r2 = client.get("/cashier/zones")
    assert r2.json()["zones"]["ROI_CASHIER"]["points"] == [[0.0, 0.0], [0.35, 1.0]]


def test_post_zones_polygon(client: TestClient, config_yaml):
    r = client.post(
        "/cashier/zones",
        json={
            "ROI_CASHIER": {
                "shape": "polygon",
                "points": [
                    {"x": 0.1, "y": 0.1},
                    {"x": 0.5, "y": 0.1},
                    {"x": 0.5, "y": 0.9},
                    {"x": 0.1, "y": 0.9},
                ],
                "active": True,
            },
        },
    )
    assert r.status_code == 200
    pts = r.json()["config"]["zones"]["ROI_CASHIER"]["points"]
    assert len(pts) == 4
    assert pts[0] == [0.1, 0.1]


def test_post_zones_validation_rejects_oob_point(client: TestClient, config_yaml):
    r = client.post(
        "/cashier/zones",
        json={
            "ROI_CASHIER": {
                "shape": "rectangle",
                "points": [{"x": 0.0, "y": 0.0}, {"x": 1.5, "y": 1.0}],
                "active": True,
            },
        },
    )
    assert r.status_code == 422


def test_post_zones_detail_config_and_detection_threshold(client: TestClient, config_yaml):
    r = client.post(
        "/cashier/zones",
        json={
            "detail_config": {"drawerOpenLimit": 15, "serviceWaitLimit": 60},
            "detection_threshold": 55,
        },
    )
    assert r.status_code == 200
    cfg = r.json()["config"]
    assert cfg["detail_config"]["drawerOpenLimit"] == 15
    assert cfg["thresholds"]["detection_threshold"] == 55


def test_post_zones_reset_writes_defaults(client: TestClient, config_yaml):
    r = client.post("/cashier/zones/reset")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "reset_to_default"
    assert "zones" in body["config"]
    assert "ROI_CASHIER" in body["config"]["zones"]
    assert body["config"]["zones"]["ROI_CASHIER"]["points"] == [[0.0, 0.0], [0.5, 1.0]]


def test_get_zones_json_config(client: TestClient, tmp_path, monkeypatch):
    path = tmp_path / "zones.json"
    path.write_text(
        json.dumps({"zones": {}, "thresholds": {"customer_wait_max_seconds": 40}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("CASHIER_CONFIG", str(path))
    r = client.get("/cashier/zones")
    assert r.status_code == 200
    assert r.json()["thresholds"]["customer_wait_max_seconds"] == 40


# ─────────────────────────────────────────────
# Status & events
# ─────────────────────────────────────────────


def test_get_status_empty(client: TestClient):
    r = client.get("/cashier/status")
    assert r.status_code == 200
    assert r.json() == {}


def test_get_status_after_push_result(client: TestClient):
    push_result(
        "cam-a",
        {
            "case_id": "N1",
            "severity": "NORMAL",
            "summary": {"cashier_zone": {"persons": 1, "drawers": 0, "cash": 0}},
        },
    )
    r = client.get("/cashier/status")
    assert r.status_code == 200
    data = r.json()
    assert "cam-a" in data
    assert data["cam-a"]["case_id"] == "N1"


def test_get_events_empty(client: TestClient):
    r = client.get("/cashier/events")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 0
    assert body["events"] == []


def test_get_events_logged_on_alert(client: TestClient):
    push_result(
        "1",
        {
            "severity": "CRITICAL",
            "case_id": "A3",
            "summary": {"alerts": ["A3 test"], "transaction": False},
        },
    )
    r = client.get("/cashier/events")
    assert r.status_code == 200
    assert r.json()["total"] == 1


def test_get_events_filter_and_pagination(client: TestClient):
    push_result(
        "1",
        {"severity": "CRITICAL", "case_id": "A3", "summary": {"alerts": ["x"]}},
    )
    push_result(
        "2",
        {"severity": "NORMAL", "case_id": "N3", "summary": {"alerts": ["y"]}},
    )
    r = client.get("/cashier/events", params={"severity": "CRITICAL"})
    assert r.json()["total"] == 1
    r2 = client.get("/cashier/events", params={"case_id": "N3"})
    assert r2.json()["total"] == 1
    r3 = client.get("/cashier/events", params={"camera_id": "2"})
    assert r3.json()["total"] == 1
    r4 = client.get("/cashier/events", params={"limit": 1, "offset": 0})
    assert len(r4.json()["events"]) == 1


def test_get_events_includes_transaction_without_alerts(client: TestClient):
    push_result(
        "1",
        {"severity": "NORMAL", "case_id": "N3", "summary": {"transaction": True, "alerts": []}},
    )
    r = client.get("/cashier/events")
    assert r.json()["total"] == 1


def test_delete_events(client: TestClient):
    push_result("1", {"summary": {"alerts": ["z"]}})
    r = client.delete("/cashier/events")
    assert r.status_code == 200
    assert r.json()["cleared"] == 1
    assert client.get("/cashier/events").json()["total"] == 0


# ─────────────────────────────────────────────
# Evidence
# ─────────────────────────────────────────────


def test_get_evidence_empty_dir(client: TestClient, evidence_dir):
    r = client.get("/cashier/evidence")
    assert r.status_code == 200
    assert r.json() == {"total": 0, "files": []}


def test_get_evidence_lists_jpg_and_filters(client: TestClient, evidence_dir):
    # API filters with `severity.lower() in f.parts` (exact path component match).
    alert_dir = evidence_dir / "alert" / "A3"
    alert_dir.mkdir(parents=True)
    jpg = alert_dir / "cam_1.jpg"
    jpg.write_bytes(b"\xff\xd8\xff fake jpeg")
    r = client.get("/cashier/evidence")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 1
    assert body["files"][0]["path"].replace("\\", "/") == "alert/A3/cam_1.jpg"

    r2 = client.get("/cashier/evidence", params={"severity": "alert", "case_id": "A3"})
    assert r2.json()["total"] == 1


def test_get_evidence_download(client: TestClient, evidence_dir):
    sub = evidence_dir / "NORMAL" / "N1"
    sub.mkdir(parents=True)
    rel = sub / "frame.jpg"
    rel.write_bytes(b"\xff\xd8\xff payload")
    r = client.get("/cashier/evidence/NORMAL/N1/frame.jpg")
    assert r.status_code == 200
    assert r.content == b"\xff\xd8\xff payload"


def test_get_evidence_download_404(client: TestClient, evidence_dir):
    r = client.get("/cashier/evidence/missing.jpg")
    assert r.status_code == 404
    err = r.json()
    assert err["status"] == "error"
    assert err["error"]["code"] == "404"


def test_get_evidence_path_traversal_forbidden(client: TestClient, evidence_dir):
    r = client.get("/cashier/evidence/../../../etc/passwd")
    assert r.status_code in (403, 404)
    if r.status_code == 403:
        assert r.json()["error"]["code"] == "403"


# ─────────────────────────────────────────────
# SSE — first event (HTTP clients block until ping without closing the stream)
# ─────────────────────────────────────────────


async def _first_sse_chunk(camera_id: str, alert_only: bool) -> str:
    gen = cashier_api._sse_publisher.stream(camera_id, alert_only=alert_only)
    try:
        return await asyncio.wait_for(gen.__anext__(), timeout=2.0)
    finally:
        await gen.aclose()


def test_sse_stream_first_event_connected():
    line = asyncio.run(_first_sse_chunk("1", alert_only=False))
    assert "event: connected" in line
    assert '"camera_id": "1"' in line


def test_sse_stream_first_event_alerts_only_flag():
    line = asyncio.run(_first_sse_chunk("1", alert_only=True))
    assert "event: connected" in line
    assert "alert_only" in line
    assert "true" in line.lower()


# ─────────────────────────────────────────────
# Media
# ─────────────────────────────────────────────


def test_media_latest_jpg_404(client: TestClient, evidence_dir):
    r = client.get("/cashier/media/1/latest/jpg")
    assert r.status_code == 404
    assert r.json()["status"] == "error"


def test_media_latest_jpg_skips_thumb(client: TestClient, evidence_dir):
    (evidence_dir / "1_main.jpg").write_bytes(b"full")
    (evidence_dir / "1_main_thumb.jpg").write_bytes(b"thumb")
    r = client.get("/cashier/media/1/latest/jpg")
    assert r.status_code == 200
    # Names matching 1_*.jpg but containing _thumb are excluded
    assert r.content == b"full"


def test_media_latest_jpg_prefers_non_thumb(client: TestClient, evidence_dir):
    """Files sorted by name; last wins — non-_thumb should be chosen if it sorts last."""
    (evidence_dir / "1_a.jpg").write_bytes(b"first")
    (evidence_dir / "1_z.jpg").write_bytes(b"last")
    r = client.get("/cashier/media/1/latest/jpg")
    assert r.status_code == 200
    assert r.content == b"last"


def test_media_latest_gif(client: TestClient, evidence_dir):
    (evidence_dir / "1_anim.gif").write_bytes(b"GIF89a")
    r = client.get("/cashier/media/1/latest/gif")
    assert r.status_code == 200
    assert r.content.startswith(b"GIF")


def test_media_latest_gif_404(client: TestClient, evidence_dir):
    r = client.get("/cashier/media/9/latest/gif")
    assert r.status_code == 404


def test_media_event_jpg(client: TestClient, evidence_dir):
    # event_id split: ts = parts[1:4] for len(parts) >= 4
    (evidence_dir / "1_2026_04_07_extra.jpg").write_bytes(b"jpgdata")
    r = client.get("/cashier/media/1/event/1_2026_04_07_12-00-00/jpg")
    assert r.status_code == 200
    assert r.content == b"jpgdata"


def test_media_event_gif(client: TestClient, evidence_dir):
    (evidence_dir / "1_2026_04_07_extra.gif").write_bytes(b"GIF89a")
    r = client.get("/cashier/media/1/event/1_2026_04_07_12-00-00/gif")
    assert r.status_code == 200


def test_media_drawer_count(client: TestClient, evidence_dir):
    logs = evidence_dir / "logs"
    logs.mkdir(parents=True)
    logf = logs / "events.jsonl"
    logf.write_text(
        '{"camera_id": "1", "status": "triggered"}\n'
        '{"camera_id": "1", "status": "idle"}\n'
        '{"camera_id": "2", "status": "triggered"}\n',
        encoding="utf-8",
    )
    r = client.get("/cashier/media/1/drawer_count")
    assert r.status_code == 200
    assert r.json() == {"camera_id": "1", "drawer_open_count": 1}
