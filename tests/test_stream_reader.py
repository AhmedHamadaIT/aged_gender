from __future__ import annotations

import numpy as np

import stream


def test_capture_preview_frame_settles_then_returns_copy(monkeypatch):
    def fake_frames(source=None, camera_id=None):
        for i in range(20):
            yield np.full((2, 2, 3), i, dtype=np.uint8)

    monkeypatch.setattr(stream, "frames", fake_frames)
    out = stream.capture_preview_frame(
        "rtsp://camera/live",
        max_reads=10,
        settle_after_reads=3,
        camera_id="cam1",
    )
    assert out is not None
    assert out.shape == (2, 2, 3)
    assert int(out[0, 0, 0]) == 2
    out[0, 0, 0] = 99
    gen = stream.frames("rtsp://camera/live", camera_id="cam1")
    assert next(gen)[0, 0, 0] == 0
    gen.close()


def test_build_rtsp_ffmpeg_options_includes_tcp(monkeypatch):
    monkeypatch.delenv("RTSP_FFMPEG_EXTRA_OPTIONS", raising=False)
    monkeypatch.delenv("RTSP_FFMPEG_OPTIONS", raising=False)
    monkeypatch.setenv("RTSP_PROFILE", "balanced")
    monkeypatch.setenv("RTSP_LOW_DELAY", "false")
    from utils import rtsp_ffmpeg

    opts = rtsp_ffmpeg.build_rtsp_ffmpeg_options()
    assert "rtsp_transport;tcp" in opts
