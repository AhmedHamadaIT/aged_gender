"""
Static geometry for FrameBus live JPEG (WebSocket / Redis preview).

**Per channel (camera):** ``build_live_stream_overlay(chan_tasks)`` receives
only tasks for that ``channelId``, so each FrameBus draws geometry for that
camera alone — multiple cameras do not share one overlay payload.

**Cross-line (``CROSS_LINE``):** every enabled task's ``areaPosition`` is
parsed; lines from *all* line tasks on the same channel are merged. Labels
include ``taskId`` when more than one line task exists on the channel so
operators can tell doors / counting lines apart.

**Cashier (``CASHIER_BOX_OPEN``):** prefer **per-task** geometry when
``areaPosition`` is a JSON object carrying ``zones`` (same shape as
``POST /cashier/zones`` / ``cashier_zones.yaml`` — normalized ``[0, 1]``
points). If no cashier task supplies usable ``areaPosition`` zones, fall
back to ``CASHIER_CONFIG`` (global file), matching today's CashierService
behaviour. Coordinates in ``areaPosition`` for cross-line remain **pixels**
in inference frame space, same as ``services/cross_line.py``.

Snapshot is taken at ``POST /detection/start`` (and watchdog respawn). Zone
file edits picked up by the cashier worker without restart are **not**
reflected on the live overlay until detection is restarted.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

_CLR_CASHIER_BGR = (0, 200, 100)
_CLR_CUSTOMER_BGR = (0, 165, 255)


def _count_enabled_cross_line_tasks(chan_tasks: List[dict]) -> int:
    n = 0
    for t in chan_tasks:
        if t.get("algorithmType") != "CROSS_LINE":
            continue
        if not t.get("enable", True):
            continue
        n += 1
    return n


def _parse_cross_lines_from_tasks(chan_tasks: List[dict]) -> List[dict]:
    out: List[dict] = []
    n_line_tasks = _count_enabled_cross_line_tasks(chan_tasks)
    for t in chan_tasks:
        if t.get("algorithmType") != "CROSS_LINE":
            continue
        if not t.get("enable", True):
            continue
        raw = t.get("areaPosition")
        if raw is None:
            continue
        if isinstance(raw, list):
            arr = raw
        else:
            try:
                arr = json.loads(str(raw))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
        if not isinstance(arr, list):
            continue
        tid = t.get("taskId")
        tname = t.get("taskName")
        prefix = ""
        if n_line_tasks > 1 and tid is not None:
            prefix = f"[{tid}] "
        elif n_line_tasks > 1 and tname:
            prefix = f"{tname}: "
        for i, obj in enumerate(arr):
            if not isinstance(obj, dict):
                continue
            pts = obj.get("point")
            if not isinstance(pts, list) or len(pts) < 2:
                continue
            a, b = pts[0], pts[1]
            if not isinstance(a, dict) or not isinstance(b, dict):
                continue
            try:
                x0, y0 = int(a["x"]), int(a["y"])
                x1, y1 = int(b["x"]), int(b["y"])
            except (KeyError, TypeError, ValueError):
                continue
            name = str(obj.get("line_name") or obj.get("line_id") or f"line_{i}")
            out.append(
                {"x0": x0, "y0": y0, "x1": x1, "y1": y1, "label": f"{prefix}{name}"}
            )
    return out


def _polys_from_zones_dict(
    zones: Dict[str, Any], *, label_suffix: str = ""
) -> List[dict]:
    """Normalized polygons for ROI_CASHIER / ROI_CUSTOMER."""
    polys: List[dict] = []
    for zone_name, color in (
        ("ROI_CASHIER", _CLR_CASHIER_BGR),
        ("ROI_CUSTOMER", _CLR_CUSTOMER_BGR),
    ):
        z = zones.get(zone_name)
        if not isinstance(z, dict):
            continue
        if not z.get("active", True):
            continue
        raw_pts = z.get("points") or []
        norm: List[List[float]] = []
        for p in raw_pts:
            if isinstance(p, dict) and "x" in p and "y" in p:
                try:
                    norm.append([float(p["x"]), float(p["y"])])
                except (TypeError, ValueError):
                    continue
            elif isinstance(p, (list, tuple)) and len(p) >= 2:
                try:
                    norm.append([float(p[0]), float(p[1])])
                except (TypeError, ValueError):
                    continue
        if len(norm) >= 2:
            label = zone_name + (f" {label_suffix}" if label_suffix else "")
            polys.append({"label": label, "points_norm": norm, "color_bgr": color})
    return polys


def _cashier_zones_dict_from_task_area_position(task: dict) -> Optional[Dict[str, Any]]:
    """
    If ``areaPosition`` embeds cashier zones (per-task), return a ``zones`` dict.
    Accepted shapes:
      * ``{"zones": {"ROI_CASHIER": {...}, ...}}``
      * ``{"ROI_CASHIER": {...}, "ROI_CUSTOMER": {...}}`` (only ROI_* keys used)
    """
    raw = task.get("areaPosition")
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s in ("[]", "{}"):
        return None
    try:
        parsed: Any = json.loads(s) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    inner = parsed.get("zones")
    if isinstance(inner, dict) and inner:
        return inner
    roi_only = {
        k: v
        for k, v in parsed.items()
        if isinstance(k, str) and k.startswith("ROI_") and isinstance(v, dict)
    }
    return roi_only if roi_only else None


def _cashier_overlay_polys_from_tasks(chan_tasks: List[dict]) -> List[dict]:
    """Merge per-task cashier zones (areaPosition); skip tasks with no zones JSON."""
    merged: List[dict] = []
    for t in chan_tasks:
        if t.get("algorithmType") != "CASHIER_BOX_OPEN":
            continue
        if not t.get("enable", True):
            continue
        zd = _cashier_zones_dict_from_task_area_position(t)
        if not zd:
            continue
        tid = t.get("taskId")
        tname = t.get("taskName")
        suffix = ""
        if tid is not None:
            suffix = f"(task {tid})"
        elif tname:
            suffix = f"({tname})"
        merged.extend(_polys_from_zones_dict(zd, label_suffix=suffix.strip()))
    return merged


def _cashier_zones_from_config() -> List[dict]:
    path = Path(os.getenv("CASHIER_CONFIG", "./config/cashier_zones.yaml"))
    if not path.is_file():
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    cfg: Dict[str, Any]
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore

            cfg = yaml.safe_load(text) or {}
        except Exception:
            return []
    else:
        try:
            cfg = json.loads(text) if text.strip() else {}
        except json.JSONDecodeError:
            return {}
    zones = cfg.get("zones") or {}
    return _polys_from_zones_dict(zones, label_suffix="")


def _channel_has_cashier_task(chan_tasks: List[dict]) -> bool:
    for t in chan_tasks:
        if t.get("algorithmType") != "CASHIER_BOX_OPEN":
            continue
        if t.get("enable", True):
            return True
    return False


def build_live_stream_overlay(chan_tasks: List[dict]) -> Optional[Dict[str, Any]]:
    """
    Return a picklable dict for FrameBus, or ``None`` if there is nothing to draw.
    """
    cross = _parse_cross_lines_from_tasks(chan_tasks)
    cashier: List[dict] = []
    if _channel_has_cashier_task(chan_tasks):
        cashier = _cashier_overlay_polys_from_tasks(chan_tasks)
        if not cashier:
            cashier = _cashier_zones_from_config()
    if not cross and not cashier:
        return None
    return {"cross_lines": cross, "cashier_zones": cashier}
