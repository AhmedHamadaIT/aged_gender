"""
utils/geometry.py
-----------------
M-9: Shared geometry primitives extracted from duplicated service implementations.

Replaces identical copies spread across:
    services/cross_line.py        — _line_side
    services/mask_hairnet_chef_hat.py — _point_in_polygon
    services/phone_usage.py       — _point_in_polygon
    services/cashier.py           — _iou, _point_in_poly, _rect_to_poly
"""

from __future__ import annotations

from typing import List, Tuple


# ── Line-side test (signed cross-product) ─────────────────────────────────────

def line_side(point: Tuple, p1: Tuple, p2: Tuple) -> int:
    """
    Return the sign of the cross-product (p2-p1) × (point-p1).
    +1 → left side, -1 → right side, 0 → collinear.
    """
    cross = (p2[0] - p1[0]) * (point[1] - p1[1]) - (p2[1] - p1[1]) * (point[0] - p1[0])
    if cross > 0:
        return 1
    if cross < 0:
        return -1
    return 0


# ── Point-in-polygon (ray casting) — {x, y} dict vertices ────────────────────

def point_in_polygon_dict(point: tuple, polygon: list) -> bool:
    """
    Ray-casting algorithm.
    ``polygon`` is a list of ``{"x": ..., "y": ...}`` dicts (used by
    MASK_HAIRNET_CHEF_HAT and PHONE_USAGE tasks).
    """
    if not polygon:
        return False
    x, y = point
    inside = False
    px, py = polygon[-1]["x"], polygon[-1]["y"]
    for pt in polygon:
        cx, cy = pt["x"], pt["y"]
        if ((cy > y) != (py > y)) and (x < (px - cx) * (y - cy) / (py - cy + 1e-9) + cx):
            inside = not inside
        px, py = cx, cy
    return inside


# ── Point-in-polygon (ray casting) — [x, y] list vertices ────────────────────

def point_in_poly(px: float, py: float, poly: List[List[float]]) -> bool:
    """
    Ray-casting point-in-polygon test.
    ``poly`` is a list of ``[x, y]`` normalised coordinate pairs
    (used by the CASHIER task).
    """
    n, inside, j = len(poly), False, len(poly) - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > py) != (yj > py)) and (
            px < (xj - xi) * (py - yi) / (yj - yi + 1e-9) + xi
        ):
            inside = not inside
        j = i
    return inside


# ── Intersection-over-Union ───────────────────────────────────────────────────

def iou(b1: List[float], b2: List[float]) -> float:
    """
    Intersection-over-Union for two ``[x1, y1, x2, y2]`` axis-aligned boxes.
    Returns 0.0 when there is no overlap or union is zero.
    """
    xi1 = max(b1[0], b2[0])
    yi1 = max(b1[1], b2[1])
    xi2 = min(b1[2], b2[2])
    yi2 = min(b1[3], b2[3])
    inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
    union = (
        (b1[2] - b1[0]) * (b1[3] - b1[1])
        + (b2[2] - b2[0]) * (b2[3] - b2[1])
        - inter
    )
    return inter / union if union > 0 else 0.0


# ── Rectangle to polygon ──────────────────────────────────────────────────────

def rect_to_poly(pts: List[List[float]]) -> List[List[float]]:
    """
    Convert a ``[[x1,y1],[x2,y2]]`` rectangle spec to a 4-corner polygon.
    """
    (x1, y1), (x2, y2) = pts[0], pts[1]
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
