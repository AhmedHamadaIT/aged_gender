"""Unit tests for live stream geometry overlay builder + FrameBus draw hook."""

from __future__ import annotations

import json

import numpy as np

from frame_bus import FrameBus
from utils.live_stream_overlay import build_live_stream_overlay


def test_two_cross_line_tasks_get_distinct_label_prefix() -> None:
    line_a = json.dumps(
        [{"line_name": "A", "point": [{"x": 0, "y": 0}, {"x": 1, "y": 1}]}]
    )
    line_b = json.dumps(
        [{"line_name": "B", "point": [{"x": 2, "y": 2}, {"x": 3, "y": 3}]}]
    )
    tasks = [
        {
            "algorithmType": "CROSS_LINE",
            "enable": True,
            "taskId": 10,
            "areaPosition": line_a,
        },
        {
            "algorithmType": "CROSS_LINE",
            "enable": True,
            "taskId": 11,
            "areaPosition": line_b,
        },
    ]
    o = build_live_stream_overlay(tasks)
    assert o is not None
    labels = {ln["label"] for ln in o["cross_lines"]}
    assert "[10] A" in labels and "[11] B" in labels


def test_cashier_zones_from_task_area_position_not_file() -> None:
    zones_body = {
        "zones": {
            "ROI_CASHIER": {
                "active": True,
                "points": [[0.0, 0.0], [0.4, 0.0], [0.4, 1.0], [0.0, 1.0]],
            },
            "ROI_CUSTOMER": {
                "active": True,
                "points": [[0.4, 0.0], [1.0, 0.0], [1.0, 1.0], [0.4, 1.0]],
            },
        }
    }
    tasks = [
        {
            "algorithmType": "CASHIER_BOX_OPEN",
            "enable": True,
            "taskId": 99,
            "taskName": "c1",
            "areaPosition": json.dumps(zones_body),
        }
    ]
    o = build_live_stream_overlay(tasks)
    assert o is not None
    assert len(o["cashier_zones"]) == 2
    lbls = " ".join(z["label"] for z in o["cashier_zones"])
    assert "task 99" in lbls


def test_parse_effective_cross_lines_matches_overlay_segment_count() -> None:
    """Invalid line objects must not produce stream segments without worker lines."""
    from services.cross_line import parse_effective_cross_lines

    raw = json.dumps(
        [
            {"line_id": "ok", "point": [{"x": 1, "y": 2}, {"x": 3, "y": 4}]},
            {"line_id": "bad", "point": [{"x": "nope", "y": 0}, {"x": 1, "y": 1}]},
        ]
    )
    assert len(parse_effective_cross_lines(raw)) == 1


def test_build_cross_line_from_json_string() -> None:
    ap = json.dumps(
        [
            {
                "line_id": "1",
                "line_name": "Door",
                "point": [{"x": 1, "y": 2}, {"x": 9, "y": 8}],
                "direction": 0,
            }
        ]
    )
    tasks = [
        {
            "algorithmType": "CROSS_LINE",
            "enable": True,
            "areaPosition": ap,
        }
    ]
    o = build_live_stream_overlay(tasks)
    assert o is not None
    assert len(o["cross_lines"]) == 1
    ln = o["cross_lines"][0]
    assert ln["x0"] == 1 and ln["y0"] == 2 and ln["x1"] == 9 and ln["y1"] == 8
    assert ln["label"] == "Door"
    assert o["cashier_zones"] == []


def test_build_nothing_when_cross_disabled_and_no_cashier() -> None:
    tasks = [
        {
            "algorithmType": "CROSS_LINE",
            "enable": False,
            "areaPosition": json.dumps(
                [{"line_id": "1", "point": [{"x": 0, "y": 0}, {"x": 1, "y": 1}]}]
            ),
        }
    ]
    assert build_live_stream_overlay(tasks) is None


def test_draw_live_stream_geometry_mutates_frame() -> None:
    bus = FrameBus.__new__(FrameBus)
    bus._live_overlay = {
        "cross_lines": [{"x0": 4, "y0": 4, "x1": 20, "y1": 20, "label": "L"}],
        "cashier_zones": [],
    }
    img = np.zeros((40, 40, 3), dtype=np.uint8)
    before = img.copy()
    bus._draw_live_stream_geometry(img)
    assert not np.array_equal(img, before)
