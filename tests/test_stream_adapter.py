"""Tests for adaptive RTSP stream probing."""

from __future__ import annotations

import json
from types import SimpleNamespace

from stream_adapter import AdaptiveStream, CameraProfile, StreamProber, StreamQuality


def test_probe_derives_4k_target_and_decoder(monkeypatch):
    payload = {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 3840,
                "height": 2160,
                "avg_frame_rate": "13/1",
            },
            {"codec_type": "audio"},
        ]
    }

    def fake_run(*_args, **_kwargs):
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setenv("RTSP_ENABLE_GPU_DECODE", "true")
    monkeypatch.delenv("STREAM_ADAPTER_WIDTH", raising=False)
    monkeypatch.delenv("STREAM_ADAPTER_HEIGHT", raising=False)
    monkeypatch.delenv("STREAM_ADAPTER_TARGET_FPS", raising=False)
    monkeypatch.setattr("subprocess.run", fake_run)

    profile = StreamProber.probe("rtsp://example/cam", camera_id="cam1")

    assert profile.quality == StreamQuality.ULTRA
    assert (profile.target_width, profile.target_height) == (640, 480)
    assert profile.target_fps == 5.0
    assert profile.hw_decoder_requested == "h264_v4l2m2m"
    assert profile.has_audio is True


def test_adaptive_stream_command_scales_and_outputs_bgr(monkeypatch):
    monkeypatch.setenv("RTSP_ENABLE_GPU_DECODE", "false")
    profile = CameraProfile(
        camera_id="cam1",
        url="rtsp://example/cam",
        target_width=640,
        target_height=480,
        target_fps=5.0,
    )
    stream = AdaptiveStream(profile)

    cmd = stream._build_ffmpeg_cmd()
    vf = cmd[cmd.index("-vf") + 1]

    assert "-vf" in cmd
    assert f"scale={profile.target_width}:{profile.target_height}:flags=fast_bilinear" in vf
    assert cmd[-3:] == ["-pix_fmt", "bgr24", "pipe:1"]
