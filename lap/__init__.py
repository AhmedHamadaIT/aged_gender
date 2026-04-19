"""
Lightweight compatibility shim for ``lap.lapjv`` used by Ultralytics tracking.

This avoids a hard runtime dependency on the compiled ``lap`` wheel on systems
where it is unavailable (for example some Jetson images). The implementation is
backed by SciPy's Hungarian solver and returns the same tuple shape that
Ultralytics expects: ``(total_cost, row_to_col, col_to_row)``.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import linear_sum_assignment


def lapjv(cost_matrix, extend_cost: bool = False, cost_limit: float = math.inf):
    cost = np.asarray(cost_matrix, dtype=float)
    if cost.ndim != 2:
        raise ValueError("cost_matrix must be 2-dimensional")

    rows, cols = cost.shape
    row_to_col = np.full(rows, -1, dtype=int)
    col_to_row = np.full(cols, -1, dtype=int)

    if rows == 0 or cols == 0:
        return 0.0, row_to_col, col_to_row

    finite_limit = math.isfinite(cost_limit)
    if extend_cost:
        dim = max(rows, cols)
        pad_value = float(cost_limit) if finite_limit else float(np.max(cost) + 1.0)
        work = np.full((dim, dim), pad_value, dtype=float)
        work[:rows, :cols] = cost
    else:
        work = cost

    matched_rows, matched_cols = linear_sum_assignment(work)

    total_cost = 0.0
    for row_idx, col_idx in zip(matched_rows, matched_cols):
        if row_idx >= rows or col_idx >= cols:
            continue
        entry_cost = float(cost[row_idx, col_idx])
        if finite_limit and entry_cost > cost_limit:
            continue
        row_to_col[row_idx] = col_idx
        col_to_row[col_idx] = row_idx
        total_cost += entry_cost

    return total_cost, row_to_col, col_to_row
