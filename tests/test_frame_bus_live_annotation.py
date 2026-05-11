"""FrameBus live annotation and publish cadence helpers (no YOLO / no camera)."""

from __future__ import annotations

import numpy as np

from frame_bus import FrameBus
from services.detector import Detection


def test_annotate_no_detections_returns_raw():
    """No detections → raw frame, no draw cost, had_boxes=False."""
    bus = FrameBus.__new__(FrameBus)
    bus._live_annotation_mode = "opencv"
    bus.camera_id = "t"
    bus._annotation_debug_every = 9999
    f = np.zeros((20, 20, 3), dtype=np.uint8)
    out, had = bus._annotate_for_stream(
        f, None, [], need_draw=True, frame_count=1
    )
    assert out is f
    assert had is False


def test_annotate_detections_always_draws_opencv_mode():
    """opencv mode + detections → annotated, had_boxes=True."""
    bus = FrameBus.__new__(FrameBus)
    bus._live_annotation_mode = "opencv"
    bus.camera_id = "t"
    bus._annotation_debug_every = 9999
    f = np.zeros((40, 40, 3), dtype=np.uint8)
    dets = [
        Detection(x1=2, y1=2, x2=10, y2=10,
                  class_id=0, class_name="person", confidence=0.9, track_id=5),
    ]
    out, had = bus._annotate_for_stream(
        f, None, dets, need_draw=True, frame_count=1
    )
    assert out.shape == f.shape
    assert had is True
    assert int(out.sum()) > 0


def test_annotate_detections_always_draws_none_mode():
    """none mode + detections → still draws (OpenCV fallback), had_boxes=True."""
    bus = FrameBus.__new__(FrameBus)
    bus._live_annotation_mode = "none"
    bus.camera_id = "t"
    bus._annotation_debug_every = 9999
    f = np.zeros((40, 40, 3), dtype=np.uint8)
    dets = [
        Detection(x1=2, y1=2, x2=10, y2=10,
                  class_id=0, class_name="person", confidence=0.9, track_id=5),
    ]
    out, had = bus._annotate_for_stream(
        f, None, dets, need_draw=True, frame_count=1
    )
    assert out.shape == f.shape
    assert had is True
    assert int(out.sum()) > 0


def test_annotate_detections_draws_even_when_need_draw_false():
    """detections > 0 overrides need_draw=False — annotated frame returned."""
    bus = FrameBus.__new__(FrameBus)
    bus._live_annotation_mode = "opencv"
    bus.camera_id = "t"
    bus._annotation_debug_every = 9999
    f = np.zeros((40, 40, 3), dtype=np.uint8)
    dets = [
        Detection(x1=2, y1=2, x2=10, y2=10,
                  class_id=0, class_name="person", confidence=0.9, track_id=5),
    ]
    out, had = bus._annotate_for_stream(
        f, None, dets, need_draw=False, frame_count=1
    )
    assert had is True
    assert int(out.sum()) > 0
