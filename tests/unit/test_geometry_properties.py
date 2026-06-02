"""
tests/unit/test_geometry_properties.py
----------------------------------------
S-8: Property-style tests for utils/geometry.py shared primitives.
"""

from __future__ import annotations

import pytest

from utils.geometry import iou, line_side, point_in_poly, point_in_polygon_dict, rect_to_poly


# ── line_side ─────────────────────────────────────────────────────────────────

class TestLineSide:
    def test_left_of_vertical(self):
        assert line_side((0, 5), (5, 0), (5, 10)) == 1

    def test_right_of_vertical(self):
        assert line_side((10, 5), (5, 0), (5, 10)) == -1

    def test_on_line(self):
        assert line_side((5, 5), (5, 0), (5, 10)) == 0

    def test_horizontal_above(self):
        # Horizontal line y=5; point y=3 is "above" (negative cross)
        result = line_side((5, 3), (0, 5), (10, 5))
        assert result in (-1, 1)  # just check it's non-zero

    @pytest.mark.parametrize("x,expected_sign", [(0, 1), (6, -1)])
    def test_symmetric(self, x, expected_sign):
        result = line_side((x, 5), (3, 0), (3, 10))
        assert result == expected_sign


# ── point_in_polygon_dict ─────────────────────────────────────────────────────

class TestPointInPolygonDict:
    _square = [
        {"x": 0, "y": 0},
        {"x": 10, "y": 0},
        {"x": 10, "y": 10},
        {"x": 0, "y": 10},
    ]

    def test_inside(self):
        assert point_in_polygon_dict((5, 5), self._square)

    def test_outside(self):
        assert not point_in_polygon_dict((15, 5), self._square)

    def test_empty_polygon(self):
        assert not point_in_polygon_dict((5, 5), [])

    def test_single_point_polygon(self):
        # Degenerate — shouldn't crash
        assert not point_in_polygon_dict((5, 5), [{"x": 5, "y": 5}])

    @pytest.mark.parametrize("pt,inside", [
        ((1, 1), True),
        ((9, 9), True),
        ((11, 5), False),
        ((5, 11), False),
    ])
    def test_parametrized(self, pt, inside):
        assert point_in_polygon_dict(pt, self._square) == inside


# ── point_in_poly ─────────────────────────────────────────────────────────────

class TestPointInPoly:
    _tri = [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]

    def test_inside_triangle(self):
        assert point_in_poly(0.5, 0.3, self._tri)

    def test_outside_triangle(self):
        assert not point_in_poly(0.9, 0.9, self._tri)

    def test_empty(self):
        assert not point_in_poly(0.5, 0.5, [])


# ── iou ───────────────────────────────────────────────────────────────────────

class TestIou:
    def test_perfect_overlap(self):
        b = [0.0, 0.0, 10.0, 10.0]
        assert iou(b, b) == pytest.approx(1.0)

    def test_no_overlap(self):
        assert iou([0, 0, 5, 5], [10, 10, 20, 20]) == pytest.approx(0.0)

    def test_half_overlap(self):
        b1 = [0.0, 0.0, 10.0, 10.0]
        b2 = [5.0, 0.0, 15.0, 10.0]
        result = iou(b1, b2)
        assert 0.30 < result < 0.36  # 50/(150) ≈ 0.333

    def test_zero_area_box(self):
        assert iou([0, 0, 0, 0], [0, 0, 5, 5]) == pytest.approx(0.0)

    @pytest.mark.parametrize("b1,b2,expected", [
        ([0,0,2,2], [1,1,3,3], pytest.approx(1/7, abs=0.01)),
        ([0,0,4,4], [0,0,2,2], pytest.approx(0.25)),
    ])
    def test_parametrized(self, b1, b2, expected):
        assert iou(b1, b2) == expected


# ── rect_to_poly ──────────────────────────────────────────────────────────────

class TestRectToPoly:
    def test_four_corners(self):
        poly = rect_to_poly([[0, 0], [10, 5]])
        assert len(poly) == 4

    def test_all_corners_present(self):
        poly = rect_to_poly([[2, 3], [8, 7]])
        xs = {p[0] for p in poly}
        ys = {p[1] for p in poly}
        assert 2 in xs and 8 in xs
        assert 3 in ys and 7 in ys
