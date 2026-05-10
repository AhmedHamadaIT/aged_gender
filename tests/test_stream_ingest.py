"""STREAM_INGEST / media_pts_ingest_enabled (no PyAV required)."""

import pytest


def test_media_pts_ingest_enabled_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from stream import media_pts_ingest_enabled

    monkeypatch.setenv("STREAM_INGEST", "pyav")
    assert media_pts_ingest_enabled() is True
    monkeypatch.setenv("STREAM_INGEST", "av")
    assert media_pts_ingest_enabled() is True
    monkeypatch.setenv("STREAM_INGEST", "opencv")
    assert media_pts_ingest_enabled() is False


def test_pyav_module_reports_availability() -> None:
    from utils.stream_pyav import pyav_available

    assert isinstance(pyav_available(), bool)
