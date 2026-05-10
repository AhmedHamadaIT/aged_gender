"""Extra stream_adapter coverage."""

from __future__ import annotations

from stream_adapter import CameraProfile, StreamQuality


def test_camera_profile_reconnects_default():
    p = CameraProfile(
        camera_id="c1",
        url="rtsp://x",
        quality=StreamQuality.HIGH,
        reconnects=3,
    )
    assert p.reconnects == 3


def test_stream_quality_members():
    assert StreamQuality.ULTRA.name == "ULTRA"
