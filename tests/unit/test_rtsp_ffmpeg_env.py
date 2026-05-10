"""RTSP FFmpeg option builder."""

from __future__ import annotations

from utils import rtsp_ffmpeg


def test_clear_transport_overrides(monkeypatch):
    rtsp_ffmpeg.set_rtsp_transport("cam1", "udp")
    rtsp_ffmpeg.clear_rtsp_transport_overrides()
    assert rtsp_ffmpeg.get_rtsp_transport("cam1") == rtsp_ffmpeg.get_rtsp_transport()


def test_build_options_contains_transport(monkeypatch):
    monkeypatch.delenv("RTSP_FFMPEG_EXTRA_OPTIONS", raising=False)
    monkeypatch.setenv("RTSP_TRANSPORT", "tcp")
    rtsp_ffmpeg.clear_rtsp_transport_overrides()
    opts = rtsp_ffmpeg.build_rtsp_ffmpeg_options("cam-x")
    assert "rtsp_transport;tcp" in opts
    assert "loglevel;error" in opts


def test_legacy_rtsp_ffmpeg_options_split(monkeypatch):
    monkeypatch.setenv(
        "RTSP_FFMPEG_OPTIONS",
        "max_delay;111|fflags;nobuffer",
    )
    monkeypatch.delenv("RTSP_FFMPEG_EXTRA_OPTIONS", raising=False)
    rtsp_ffmpeg.clear_rtsp_transport_overrides()
    opts = rtsp_ffmpeg.build_rtsp_ffmpeg_options(None)
    assert "max_delay;111" in opts


def test_normalize_rtsp_source_url_strips_leading_slashes():
    raw = "//rtsp://admin:pass@10.0.0.1/stream"
    assert rtsp_ffmpeg.normalize_rtsp_source_url(raw) == "rtsp://admin:pass@10.0.0.1/stream"
    assert rtsp_ffmpeg.normalize_rtsp_source_url("  ///rtsp://h/x  ") == "rtsp://h/x"
    assert rtsp_ffmpeg.normalize_rtsp_source_url("rtsp://ok") == "rtsp://ok"
