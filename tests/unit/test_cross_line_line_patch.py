"""Cross-line line patch helper and task registry handler."""

from __future__ import annotations

import json

import pytest
from fastapi import HTTPException

from apis.tasks import CrossLineLinePatch, TaskRegistry
from services.cross_line import update_line_in_area_position


def _area_with_line(y: int = 400) -> str:
    return json.dumps(
        [
            {
                "line_id": "entrance",
                "line_name": "Entrance",
                "point": [{"x": 100, "y": y}, {"x": 900, "y": y}],
                "direction": 1,
            }
        ]
    )


def test_update_line_in_area_position_moves_y():
    updated = update_line_in_area_position(
        _area_with_line(400),
        "entrance",
        point=[{"x": 120, "y": 500}, {"x": 880, "y": 500}],
    )
    parsed = json.loads(updated)
    assert parsed[0]["point"][0]["y"] == 500
    assert parsed[0]["point"][1]["y"] == 500


def test_update_line_by_line_name():
    updated = update_line_in_area_position(
        _area_with_line(),
        "Entrance",
        direction=2,
    )
    assert json.loads(updated)[0]["direction"] == 2


def test_update_line_missing_raises():
    with pytest.raises(ValueError, match="not found"):
        update_line_in_area_position(_area_with_line(), "missing", direction=0)


def test_task_registry_patch_cross_line_updates_task(monkeypatch):
    reg = TaskRegistry()
    reg.upsert(
        __import__("apis.tasks", fromlist=["TaskConfig"]).TaskConfig(
            taskId=7,
            taskName="line_task",
            algorithmType="CROSS_LINE",
            channelId="cam1",
            enable=True,
            areaPosition=_area_with_line(300),
        )
    )
    monkeypatch.setattr(
        "apis.tasks._refresh_cross_line_stream",
        lambda task_id, task: {"applied": False, "reason": "test"},
    )
    out = reg.on_patch_cross_line(
        7,
        "entrance",
        CrossLineLinePatch(
            point=[
                {"x": 10, "y": 200},
                {"x": 990, "y": 200},
            ]
        ),
    )
    assert out["status"] == "updated"
    assert out["line_id"] == "entrance"
    saved = reg.get(7)
    assert saved is not None
    pts = json.loads(saved["areaPosition"])[0]["point"]
    assert pts[0]["y"] == 200


def test_task_registry_patch_rejects_non_cross_line():
    reg = TaskRegistry()
    reg.upsert(
        __import__("apis.tasks", fromlist=["TaskConfig"]).TaskConfig(
            taskId=8,
            taskName="ppe",
            algorithmType="MASK_HAIRNET_CHEF_HAT",
            channelId="cam1",
            enable=True,
            areaPosition="[]",
        )
    )
    with pytest.raises(HTTPException) as ei:
        reg.on_patch_cross_line(8, "1", CrossLineLinePatch(direction=0))
    assert ei.value.status_code == 400
