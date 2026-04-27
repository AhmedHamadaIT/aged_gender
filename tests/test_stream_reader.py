from __future__ import annotations

import subprocess

import numpy as np

import stream


def test_ffmpeg_options_are_merged_without_clobbering_runtime_values():
    merged = stream._merge_ffmpeg_capture_options(
        "rtsp_transport;udp|timeout;9000000|user_agent;camera-client"
    )

    assert "rtsp_transport;udp" in merged
    assert "timeout;9000000" in merged
    assert "reconnect;1" in merged
    assert "reconnect_delay_max;5" in merged
    assert "user_agent;camera-client" in merged


def test_gstreamer_h265_pipeline_uses_zero_latency_appsink(monkeypatch):
    monkeypatch.setenv("STREAM_GST_LATENCY_MS", "0")
    pipeline = stream._build_gstreamer_pipeline("rtsp://camera/live", "h265")

    assert "rtph265depay" in pipeline
    assert "h265parse" in pipeline
    assert "nvv4l2decoder" in pipeline
    assert "appsink drop=true max-buffers=1 sync=false" in pipeline


def test_ffprobe_timeout_falls_back_to_h264(monkeypatch):
    def _timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="ffprobe", timeout=kwargs["timeout"])

    monkeypatch.setenv("STREAM_CODEC", "auto")
    monkeypatch.setenv("STREAM_CODEC_FALLBACK", "h264")
    monkeypatch.setenv("STREAM_FFPROBE_TIMEOUT_SEC", "3")
    monkeypatch.setattr(stream.shutil, "which", lambda name: "/usr/bin/ffprobe")
    monkeypatch.setattr(stream.subprocess, "run", _timeout)

    assert stream._detect_rtsp_codec("rtsp://camera/live") == "h264"


def test_capture_preview_frame_settles_then_returns_copy(monkeypatch):
    def fake_frames(source=None):
        for i in range(20):
            yield np.full((2, 2, 3), i, dtype=np.uint8)

    monkeypatch.setattr(stream, "frames", fake_frames)
    out = stream.capture_preview_frame(
        "rtsp://camera/live",
        max_reads=10,
        settle_after_reads=3,
    )
    assert out is not None
    assert out.shape == (2, 2, 3)
    assert int(out[0, 0, 0]) == 2
    out[0, 0, 0] = 99
    gen = stream.frames("rtsp://camera/live")
    assert next(gen)[0, 0, 0] == 0
    gen.close()


def test_rtsp_reader_returns_frame_copy():
    reader = stream._RTSPReader("rtsp://camera/live")
    original = np.zeros((2, 2, 3), dtype=np.uint8)
    original[0, 0, 0] = 10

    with reader._lock:
        reader._latest_frame = original

    frame = reader.read_frame()
    frame[0, 0, 0] = 99

    with reader._lock:
        assert reader._latest_frame[0, 0, 0] == 10
