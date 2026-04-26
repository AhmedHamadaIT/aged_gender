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


def test_probe_hevc_respects_rtsp_jetson_hevc_decoder_override(monkeypatch):
    payload = {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "hevc",
                "width": 1920,
                "height": 1080,
                "avg_frame_rate": "25/1",
            },
        ]
    }

    def fake_run(*_args, **_kwargs):
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setenv("RTSP_ENABLE_GPU_DECODE", "true")
    monkeypatch.setenv("RTSP_JETSON_HEVC_DECODER", "hevc_nvmpi")
    monkeypatch.delenv("RTSP_HWDECODER", raising=False)
    monkeypatch.delenv("RTSP_JETSON_H264_DECODER", raising=False)
    monkeypatch.setattr("subprocess.run", fake_run)

    profile = StreamProber.probe("rtsp://example/cam", camera_id="cam1")
    assert profile.hw_decoder_requested == "hevc_nvmpi"
    assert profile.decoder == "hevc_nvmpi"


def test_probe_h264_respects_rtsp_hwdecoder_legacy(monkeypatch):
    payload = {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1280,
                "height": 720,
                "avg_frame_rate": "15/1",
            },
        ]
    }

    def fake_run(*_args, **_kwargs):
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setenv("RTSP_ENABLE_GPU_DECODE", "true")
    monkeypatch.setenv("RTSP_HWDECODER", "h264_nvmpi")
    monkeypatch.delenv("RTSP_JETSON_H264_DECODER", raising=False)
    monkeypatch.setattr("subprocess.run", fake_run)

    profile = StreamProber.probe("rtsp://example/cam", camera_id="cam2")
    assert profile.hw_decoder_requested == "h264_nvmpi"


def test_jetson_hevc_per_codec_override_wins_over_rtsp_hwdecoder(monkeypatch):
    payload = {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "hevc",
                "width": 1280,
                "height": 720,
                "avg_frame_rate": "12/1",
            },
        ]
    }

    def fake_run(*_args, **_kwargs):
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setenv("RTSP_ENABLE_GPU_DECODE", "true")
    monkeypatch.setenv("RTSP_JETSON_HEVC_DECODER", "hevc_primary")
    monkeypatch.setenv("RTSP_HWDECODER", "hevc_fallback")
    monkeypatch.setattr("subprocess.run", fake_run)

    profile = StreamProber.probe("rtsp://example/cam", camera_id="cam3")
    assert profile.hw_decoder_requested == "hevc_primary"


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
    assert "-timeout" not in cmd
    assert f"scale={profile.target_width}:{profile.target_height}:flags=fast_bilinear" in vf
    assert cmd[-3:] == ["-pix_fmt", "bgr24", "pipe:1"]


def test_hw_decoder_disable_detects_device_errors():
    profile = CameraProfile(
        camera_id="cam1",
        url="rtsp://example/cam",
        target_width=640,
        target_height=480,
        target_fps=5.0,
        hw_decoder_requested="hevc_v4l2m2m",
        decoder="hevc_v4l2m2m",
        hw_decoder_active=True,
    )
    stream = AdaptiveStream(profile)
    stream._stderr_tail.extend(
        b"[hevc_v4l2m2m] Could not find a valid device\n"
        b"[hevc_v4l2m2m] can't configure decoder\n"
        b"Error while opening decoder for input stream #0:0 : Invalid argument\n"
    )
    assert stream.should_disable_hw_decoder() is True


def test_hw_decoder_disable_ignores_non_device_errors():
    profile = CameraProfile(
        camera_id="cam1",
        url="rtsp://example/cam",
        target_width=640,
        target_height=480,
        target_fps=5.0,
        hw_decoder_requested="hevc_v4l2m2m",
        decoder="hevc_v4l2m2m",
        hw_decoder_active=True,
    )
    stream = AdaptiveStream(profile)
    stream._stderr_tail.extend(b"[hevc] PPS id out of range: 0\nCould not find ref with POC 12\n")
    assert stream.should_disable_hw_decoder() is False
