"""FrameBus live annotation and publish cadence helpers (no YOLO / no camera)."""

from __future__ import annotations

import numpy as np

from frame_bus import FrameBus
from services.detector import Detection


def test_annotate_skips_draw_when_not_needed():
    bus = FrameBus.__new__(FrameBus)
    bus._live_annotation_mode = "ultralytics"
    f = np.zeros((20, 20, 3), dtype=np.uint8)
    out = bus._annotate_for_stream(f, None, [], need_draw=False)
    assert out is f


def test_annotate_none_mode_passthrough():
    bus = FrameBus.__new__(FrameBus)
    bus._live_annotation_mode = "none"
    f = np.zeros((20, 20, 3), dtype=np.uint8)
    out = bus._annotate_for_stream(f, None, [], need_draw=True)
    assert out is f


def test_annotate_opencv_draws_boxes():
    bus = FrameBus.__new__(FrameBus)
    bus._live_annotation_mode = "opencv"
    f = np.zeros((40, 40, 3), dtype=np.uint8)
    dets = [
        Detection(
            x1=2, y1=2, x2=10, y2=10,
            class_id=0, class_name="person", confidence=0.9, track_id=5,
        ),
    ]
    out = bus._annotate_for_stream(f, None, dets, need_draw=True)
    assert out.shape == f.shape
    # Should have drawn non-zero pixels on the rectangle border / text region
    assert int(out.sum()) > 0
