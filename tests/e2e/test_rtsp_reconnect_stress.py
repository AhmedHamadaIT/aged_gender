"""
Stress-oriented checks for stream reconnect policy (no live RTSP required).
"""

from stream import STREAM_RECONNECT_BASE_SEC, STREAM_RECONNECT_MAX_SEC, _reconnect_delay_seconds


def test_reconnect_delay_has_jitter_bounds():
    delays = [_reconnect_delay_seconds(2) for _ in range(30)]
    assert min(delays) >= STREAM_RECONNECT_BASE_SEC * 0.2
    assert max(delays) <= STREAM_RECONNECT_MAX_SEC * 1.1