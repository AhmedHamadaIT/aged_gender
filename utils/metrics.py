"""
utils/metrics.py
----------------
M-12: Prometheus metrics registry for the Vision Pipeline API.

Metrics are gated behind PROMETHEUS_ENABLED=true (default false) so
deployments that don't use Prometheus pay zero overhead.

Exposed at GET /metrics (plain text) — completely separate from the
existing /stream/metrics JSON endpoint which is unchanged.

Usage:
    from utils.metrics import (
        active_cameras,
        frames_processed_total,
        stream_fps,
        task_queue_depth,
        inc_frames,
        observe_fps,
        set_active_cameras,
        set_queue_depth,
    )
"""

from __future__ import annotations

import os

_ENABLED: bool = os.getenv("PROMETHEUS_ENABLED", "false").lower() in ("true", "1", "yes")

if _ENABLED:
    from prometheus_client import Counter, Gauge, CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST

    _registry = CollectorRegistry()

    active_cameras = Gauge(
        "vision_active_cameras",
        "Number of camera pipelines currently running",
        registry=_registry,
    )

    frames_processed_total = Counter(
        "vision_frames_processed_total",
        "Total frames processed by the inference pipeline",
        ["camera_id"],
        registry=_registry,
    )

    stream_fps = Gauge(
        "vision_stream_fps",
        "Current measured FPS for a camera stream",
        ["camera_id"],
        registry=_registry,
    )

    task_queue_depth = Gauge(
        "vision_task_queue_depth",
        "Current number of items in a task queue",
        ["camera_id", "task_id"],
        registry=_registry,
    )

    events_emitted_total = Counter(
        "vision_events_emitted_total",
        "Total task events emitted",
        ["camera_id", "algorithm"],
        registry=_registry,
    )

    def inc_frames(camera_id: str, n: int = 1) -> None:
        frames_processed_total.labels(camera_id=camera_id).inc(n)

    def observe_fps(camera_id: str, fps: float) -> None:
        stream_fps.labels(camera_id=camera_id).set(fps)

    def set_active_cameras(count: int) -> None:
        active_cameras.set(count)

    def set_queue_depth(camera_id: str, task_id: str, depth: int) -> None:
        task_queue_depth.labels(camera_id=camera_id, task_id=task_id).set(depth)

    def inc_events(camera_id: str, algorithm: str, n: int = 1) -> None:
        events_emitted_total.labels(camera_id=camera_id, algorithm=algorithm).inc(n)

    def metrics_response() -> tuple[bytes, str]:
        """Return (body_bytes, content_type) for the /metrics endpoint."""
        return generate_latest(_registry), CONTENT_TYPE_LATEST

else:
    # Stub everything so imports never fail even when Prometheus is disabled.
    class _Stub:
        def labels(self, **_): return self
        def inc(self, _=1): pass
        def set(self, _): pass
        def observe(self, _): pass

    active_cameras = _Stub()
    frames_processed_total = _Stub()
    stream_fps = _Stub()
    task_queue_depth = _Stub()
    events_emitted_total = _Stub()

    def inc_frames(camera_id: str, n: int = 1) -> None: pass
    def observe_fps(camera_id: str, fps: float) -> None: pass
    def set_active_cameras(count: int) -> None: pass
    def set_queue_depth(camera_id: str, task_id: str, depth: int) -> None: pass
    def inc_events(camera_id: str, algorithm: str, n: int = 1) -> None: pass

    def metrics_response() -> tuple[bytes, str]:
        return b"# PROMETHEUS_ENABLED is not set\n", "text/plain; version=0.0.4"
