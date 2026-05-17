"""Cross-line + AgeGender: loading behaviour when detailConfig.enableAttrDetect is true."""

from __future__ import annotations

import json
import os
import pathlib

import pytest

# Minimal valid areaPosition for tests that don't care about geometry
_VALID_AREA = json.dumps(
    [{"line_id": "1", "line_name": "L",
      "point": [{"x": 0, "y": 0}, {"x": 100, "y": 0}], "direction": 0}]
)

ROOT = pathlib.Path(__file__).resolve().parents[2]
_ONNX = ROOT / "models" / "best_aged_gender_6.onnx"
_MIN = 512


def _ensure_task_storage_dirs() -> None:
    monkeypatch_dirs = (
        (ROOT / "artifacts" / "tmp_events_unit_ag"),
        (ROOT / "artifacts" / "tmp_captures_unit_ag"),
        (ROOT / "artifacts" / "tmp_scenes_unit_ag"),
    )
    for d in monkeypatch_dirs:
        d.mkdir(parents=True, exist_ok=True)


@pytest.mark.skipif(not _ONNX.is_file(), reason=f"missing {_ONNX}")
def test_cross_line_loads_age_gender_when_enable_attr_and_model_valid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    try:
        sz = _ONNX.stat().st_size
    except OSError:
        pytest.skip(f"cannot stat {_ONNX}")
    if sz < _MIN:
        pytest.skip(f"best_aged_gender_6.onnx is placeholder/empty ({sz} bytes); install real weights")

    _ensure_task_storage_dirs()
    monkeypatch.setenv("AGE_GENDER_MODEL", str(_ONNX))
    monkeypatch.setenv("ONNX_EXECUTION_PROVIDERS_ORDER", "cpu_only")
    monkeypatch.setenv("EVENTS_DIR", str(ROOT / "artifacts" / "tmp_events_unit_ag"))
    monkeypatch.setenv("CAPTURE_DIR", str(ROOT / "artifacts" / "tmp_captures_unit_ag"))
    monkeypatch.setenv("SCENE_DIR", str(ROOT / "artifacts" / "tmp_scenes_unit_ag"))

    from services.cross_line import CrossLineTask

    t = CrossLineTask(
        {
            "taskId": 70001,
            "taskName": "unit_crossline_ag",
            "channelId": "cam_x",
            "enable": True,
            "threshold": 50,
            "areaPosition": "[]",
            "detailConfig": {"enableAttrDetect": True, "enableReid": False},
            "validWeekday": [
                "MONDAY",
                "TUESDAY",
                "WEDNESDAY",
                "THURSDAY",
                "FRIDAY",
                "SATURDAY",
                "SUNDAY",
            ],
            "validStartTime": 0,
            "validEndTime": 86400000,
        }
    )
    assert t._age_gender is not None, "AgeGenderService should load when enableAttrDetect is true"
    assert t._age_gender_load_error is None, t._age_gender_load_error


@pytest.mark.skipif(not _ONNX.is_file(), reason=f"missing {_ONNX}")
def test_cross_line_attr_enabled_but_placeholder_model_fails_cleanly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    try:
        sz = _ONNX.stat().st_size
    except OSError:
        pytest.skip(f"cannot stat {_ONNX}")
    if sz >= _MIN:
        pytest.skip("real ONNX present — use test_cross_line_loads_age_gender_when_enable_attr_and_model_valid")

    _ensure_task_storage_dirs()
    monkeypatch.setenv("AGE_GENDER_MODEL", str(_ONNX))
    monkeypatch.setenv("ONNX_EXECUTION_PROVIDERS_ORDER", "cpu_only")
    monkeypatch.setenv("EVENTS_DIR", str(ROOT / "artifacts" / "tmp_events_unit_ag"))
    monkeypatch.setenv("CAPTURE_DIR", str(ROOT / "artifacts" / "tmp_captures_unit_ag"))
    monkeypatch.setenv("SCENE_DIR", str(ROOT / "artifacts" / "tmp_scenes_unit_ag"))

    from services.cross_line import CrossLineTask

    t = CrossLineTask(
        {
            "taskId": 70003,
            "taskName": "unit_crossline_ag_placeholder",
            "channelId": "cam_x",
            "enable": True,
            "threshold": 50,
            "areaPosition": _VALID_AREA,
            "detailConfig": {"enableAttrDetect": True, "enableReid": False},
            "validWeekday": [
                "MONDAY",
                "TUESDAY",
                "WEDNESDAY",
                "THURSDAY",
                "FRIDAY",
                "SATURDAY",
                "SUNDAY",
            ],
            "validStartTime": 0,
            "validEndTime": 86400000,
        }
    )
    assert t._age_gender is None
    assert t._age_gender_load_error is not None
    assert "too small" in (t._age_gender_load_error or "").lower()


def test_cross_line_skips_age_gender_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    for sub in ("tmp_events_unit2", "tmp_captures_unit2", "tmp_scenes_unit2"):
        (ROOT / "artifacts" / sub).mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("EVENTS_DIR", str(ROOT / "artifacts" / "tmp_events_unit2"))
    monkeypatch.setenv("CAPTURE_DIR", str(ROOT / "artifacts" / "tmp_captures_unit2"))
    monkeypatch.setenv("SCENE_DIR", str(ROOT / "artifacts" / "tmp_scenes_unit2"))

    from services.cross_line import CrossLineTask

    t = CrossLineTask(
        {
            "taskId": 70002,
            "taskName": "unit_crossline_no_ag",
            "channelId": "cam_x",
            "enable": True,
            "threshold": 50,
            "areaPosition": _VALID_AREA,
            "detailConfig": {"enableAttrDetect": False},
            "validWeekday": [
                "MONDAY",
                "TUESDAY",
                "WEDNESDAY",
                "THURSDAY",
                "FRIDAY",
                "SATURDAY",
                "SUNDAY",
            ],
            "validStartTime": 0,
            "validEndTime": 86400000,
        }
    )
    assert t._age_gender is None
