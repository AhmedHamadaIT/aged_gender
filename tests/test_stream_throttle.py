"""Tests for stream frame throttling."""

from __future__ import annotations


def test_frame_throttle_skips_every_nth_frame():
    from stream import _FrameThrottle

    throttle = _FrameThrottle(frame_skip=2)

    assert [throttle.should_yield() for _ in range(4)] == [
        False,
        True,
        False,
        True,
    ]


def test_frame_throttle_limits_target_fps():
    from stream import _FrameThrottle

    ticks = iter([0.0, 0.1, 0.25, 0.5])
    throttle = _FrameThrottle(target_fps=5, time_fn=lambda: next(ticks))

    assert [throttle.should_yield() for _ in range(4)] == [
        True,
        False,
        True,
        True,
    ]
