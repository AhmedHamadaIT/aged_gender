"""Unit tests for Jetson GStreamer pipeline strings (no camera / no OpenCV hardware)."""

from __future__ import annotations

import pytest

from stream_gstreamer import build_gstreamer_pipeline, should_attempt_gstreamer


def test_build_gstreamer_pipeline_h264(monkeypatch):
    monkeypatch.setenv("RTSP_TRANSPORT", "tcp")
    monkeypatch.setenv("RTSP_GST_LATENCY_MS", "0")
    p = build_gstreamer_pipeline("rtsp://x/cam", "h264", 1280, 720)
    assert "rtph264depay ! h264parse" in p
    assert "nvv4l2decoder" in p
    assert "width=1280" in p
    assert "height=720" in p
    assert "appsink drop=true sync=false max-buffers=1" in p
    assert 'location="rtsp://x/cam"' in p


def test_build_gstreamer_pipeline_hevc(monkeypatch):
    monkeypatch.setenv("RTSP_TRANSPORT", "tcp")
    p = build_gstreamer_pipeline("rtsp://y/main", "hevc", 640, 480)
    assert "rtph265depay ! h265parse" in p
    assert "nvv4l2decoder" in p


def test_build_gstreamer_pipeline_unsupported_codec():
    with pytest.raises(ValueError, match="unsupported"):
        build_gstreamer_pipeline("rtsp://z", "mjpeg", 640, 480)


def test_should_attempt_gstreamer_respects_mode(monkeypatch):
    import stream_gstreamer as sg

    monkeypatch.setenv("RTSP_GSTREAMER", "true")
    monkeypatch.setattr(sg, "is_jetson", lambda: True)
    monkeypatch.setattr(sg, "gstreamer_backend_available", lambda: True)
    assert sg.should_attempt_gstreamer("gstreamer") is True
    assert sg.should_attempt_gstreamer("auto") is True
    assert sg.should_attempt_gstreamer("ffmpeg") is False
    assert sg.should_attempt_gstreamer("opencv") is False
