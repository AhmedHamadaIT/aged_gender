"""Unit tests for utils.frame_flow (gate, hybrid clock, slot mailbox)."""

import time

import numpy as np
import pytest

from utils.frame_flow import (
    AdaptiveGateTuner,
    FlowMetrics,
    HybridClock,
    ScheduledFrameGate,
    SlotMailbox,
    TokenBucketFairness,
    slot_index,
)


def test_slot_index_boundary():
    origin = 0.0
    T = 0.05
    assert slot_index(0.049, origin, T) == 0
    assert slot_index(0.051, origin, T) == 1


def test_hybrid_clock_pts_backward_falls_back_mono():
    c = HybridClock(pts_jump_sec=5.0)
    t0, m0 = c.t_capture(100.0)
    assert m0 == "pts"
    t1, m1 = c.t_capture(99.0)
    assert m1 == "mono"
    assert c.pts_anomalies >= 1


def test_gate_missed_slots_catch_up():
    g = ScheduledFrameGate(20.0, ceiling_fps=30.0)
    g.accept(0.0)
    # jump far ahead
    ok = g.accept(1.0)
    assert ok
    assert g.missed_slots > 0


def test_mailbox_same_slot_overwrite():
    g = ScheduledFrameGate(100.0, ceiling_fps=100.0)  # very fast for test
    g.accept(0.0)
    mb = SlotMailbox(g, epoch_getter=lambda: 0)
    f0 = np.zeros((2, 2, 3), dtype=np.uint8)
    f1 = np.ones((2, 2, 3), dtype=np.uint8)
    assert mb.offer(f0, 0.0, "mono", time.monotonic()) == "ok"
    r = mb.offer(f1, 0.0001, "mono", time.monotonic())
    assert r == "overwrite"
    got = mb.get(timeout=1.0)
    assert np.allclose(got[0], f1)


def test_mailbox_cross_slot_reject_restores_pending():
    g = ScheduledFrameGate(10.0, ceiling_fps=10.0)
    g.accept(0.0)
    mb = SlotMailbox(g, epoch_getter=lambda: 0)
    period = g.period
    f0 = np.zeros((2, 2, 3), dtype=np.uint8)
    f1 = np.ones((2, 2, 3), dtype=np.uint8)
    mb.offer(f0, 0.0, "mono", time.monotonic())
    r = mb.offer(f1, period * 1.5, "mono", time.monotonic())
    assert r == "cross_slot_reject"
    got = mb.get(timeout=1.0)
    assert np.allclose(got[0], f0)


def test_token_bucket_zero_rate_always_admits():
    b = TokenBucketFairness(0.0)
    assert b.admit() and b.admit() and b.admit()


def test_token_bucket_burst_capacity_at_least_2x_rate():
    b = TokenBucketFairness(10.0, capacity=1.0, burst_multiplier=2.0)
    assert b.capacity >= 20.0


def test_hybrid_clock_mono_holddown_after_pts_jump():
    c = HybridClock(pts_jump_sec=1.0, mono_holddown_sec=10.0)
    t0, m0 = c.t_capture(100.0)
    assert m0 == "pts"
    _, m1 = c.t_capture(200.0)
    assert m1 == "mono"
    t2, m2 = c.t_capture(None)
    assert m2 == "mono"


def test_hybrid_clock_pts_recovery_requires_streak():
    c = HybridClock(
        pts_jump_sec=100.0,
        mono_holddown_sec=0.0,
        recovery_pts_streak=3,
        backwards_eps=0.01,
    )
    assert c.t_capture(10.0)[1] == "pts"
    assert c.t_capture(9.0)[1] == "mono"
    assert c.t_capture(11.0)[1] == "mono"
    assert c.t_capture(12.0)[1] == "mono"
    assert c.t_capture(13.0)[1] == "pts"


def test_gate_min_effective_fps_floor():
    g = ScheduledFrameGate(20.0, ceiling_fps=30.0, min_effective_fps=8.0)
    g.set_effective_fps(2.0)
    assert g.effective_fps == pytest.approx(8.0)


def test_token_bucket_warm_start_on_reset():
    b = TokenBucketFairness(10.0, capacity=20.0, warm_start_seconds=0.5)
    assert b.tokens == pytest.approx(20.0)
    b.reset()
    assert b.tokens == pytest.approx(5.0)


def test_mailbox_epoch_stamped_at_offer():
    epoch = [0]
    g = ScheduledFrameGate(100.0, ceiling_fps=100.0)
    g.accept(0.0)
    mb = SlotMailbox(g, epoch_getter=lambda: epoch[0])
    f0 = np.zeros((2, 2, 3), dtype=np.uint8)
    mb.offer(f0, 0.0, "mono", time.monotonic())
    epoch[0] = 1
    frame, _, _, _, ep = mb.get(timeout=1.0)
    assert ep == 0
    assert np.allclose(frame, f0)


def test_flow_metrics_timestamp_mode_switch_count():
    m = FlowMetrics()
    m.set_timestamp_mode("mono")
    assert m.snapshot()["timestamp_mode_switch_count"] == 0
    m.set_timestamp_mode("pts")
    assert m.snapshot()["timestamp_mode_switch_count"] == 1
    m.set_timestamp_mode("pts")
    assert m.snapshot()["timestamp_mode_switch_count"] == 1
    m.set_timestamp_mode("mono")
    assert m.snapshot()["timestamp_mode_switch_count"] == 2


def test_flow_metrics_ingest_backend_switch_count():
    m = FlowMetrics()
    m.note_ingest_backend("pyav")
    assert m.snapshot()["timestamp_mode_switch_count"] == 0
    m.note_ingest_backend("opencv_ffmpeg")
    assert m.snapshot()["timestamp_mode_switch_count"] == 1


def test_adaptive_tuner_damping_limits_step():
    g = ScheduledFrameGate(20.0, ceiling_fps=30.0)
    tuner = AdaptiveGateTuner(
        g,
        adapt_every_n=1,
        e2e_p95_high_ms=50.0,
        e2e_p95_low_ms=99999.0,
        missed_window_high=999,
        max_down_step=0.85,
        max_up_step=1.05,
        input_gap_var_high_ms2=1e12,
        input_gap_var_low_ms2=1e12,
        input_gap_mad_high_ms=1e9,
        input_gap_mad_low_ms=1e9,
    )
    assert tuner.adapt_every_n == 1
    for _ in range(8):
        tuner.observe(1.0, 100.0)
    assert g.effective_fps == pytest.approx(18.0, rel=0.02)
