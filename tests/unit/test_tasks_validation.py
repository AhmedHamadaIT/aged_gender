"""TaskRegistry validation rules."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from apis.tasks import TaskConfig, TaskRegistry


def test_cross_line_empty_area_position_raises():
    reg = TaskRegistry()
    cfg = TaskConfig(
        taskId=1,
        taskName="t",
        algorithmType="CROSS_LINE",
        channelId="1",
        enable=True,
        areaPosition="",
    )
    with pytest.raises(HTTPException) as ei:
        reg.upsert(cfg)
    assert ei.value.status_code == 400


def test_cross_line_invalid_json_raises():
    reg = TaskRegistry()
    cfg = TaskConfig(
        taskId=1,
        taskName="t",
        algorithmType="CROSS_LINE",
        channelId="1",
        enable=True,
        areaPosition="not-json",
    )
    with pytest.raises(HTTPException) as ei:
        reg.upsert(cfg)
    assert ei.value.status_code == 400


def test_unsupported_algorithm_raises():
    reg = TaskRegistry()
    cfg = TaskConfig(
        taskId=2,
        taskName="x",
        algorithmType="UNKNOWN_ALG",
        channelId="1",
        enable=True,
    )
    with pytest.raises(HTTPException) as ei:
        reg.upsert(cfg)
    assert ei.value.status_code == 400


def test_cross_line_disabled_skips_area_validation():
    reg = TaskRegistry()
    cfg = TaskConfig(
        taskId=3,
        taskName="off",
        algorithmType="CROSS_LINE",
        channelId="1",
        enable=False,
        areaPosition="",
    )
    reg.upsert(cfg)
