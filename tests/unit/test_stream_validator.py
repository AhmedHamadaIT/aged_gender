"""
tests/unit/test_stream_validator.py
------------------------------------
Unit tests for ``services.stream_validator.StreamValidator``.

Each of the 7 in-process validation stages is tested in isolation.
No real camera, RTSP stream, Redis, or subprocess is needed.

All tests pass without any external services.
"""

from __future__ import annotations

import json
import socket
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from services.stream_validator import StreamValidator, _probe_stream_url, _validate_task_config
from utils.error_codes import (
    STREAM_NOT_REGISTERED,
    STREAM_UNREACHABLE,
    NO_TASK_ASSIGNED,
    TASK_NOT_FOUND,
    TASK_DISABLED,
    CAMERA_TASK_MISMATCH,
    TASK_CONFIG_INVALID,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures / helpers
# ─────────────────────────────────────────────────────────────────────────────

_VALID_LINE = json.dumps(
    [{"line_id": "1", "line_name": "L1", "point": [{"x": 0, "y": 0}, {"x": 100, "y": 0}], "direction": 0}]
)

_TASK_BASE = {
    "taskId": 1,
    "taskName": "entrance",
    "algorithmType": "CROSS_LINE",
    "channelId": "cam1",
    "enable": True,
    "threshold": 50,
    "areaPosition": _VALID_LINE,
}


def _make_cam_registry(cameras: Dict[str, str]):
    reg = MagicMock()
    reg.all.return_value = cameras
    return reg


def _make_task_registry(tasks: List[dict]):
    reg = MagicMock()
    reg.get.side_effect = lambda tid: next(
        (t for t in tasks if t["taskId"] == tid), None
    )
    reg.get_enabled.return_value = [t for t in tasks if t.get("enable", True)]
    return reg


_SENTINEL = object()

def _make_validator(cameras=_SENTINEL, tasks=None) -> StreamValidator:
    cams = {"cam1": "rtsp://192.168.1.1:554/stream"} if cameras is _SENTINEL else cameras
    task_list = tasks if tasks is not None else [_TASK_BASE.copy()]
    return StreamValidator(_make_cam_registry(cams), _make_task_registry(task_list))


def _one_task(**overrides) -> List[dict]:
    t = _TASK_BASE.copy()
    t.update(overrides)
    return [t]


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: stream registered
# ─────────────────────────────────────────────────────────────────────────────

class TestStage1StreamRegistered:
    def test_fails_when_camera_not_registered(self):
        v = _make_validator(cameras={})
        f = v.validate_channel("cam1", "rtsp://x:554/s", _one_task())
        assert f is not None
        assert f.stage == 1
        assert f.error_code == STREAM_NOT_REGISTERED
        assert "cam1" in f.message

    def test_passes_when_camera_registered(self):
        with patch("services.stream_validator._PROBE_ENABLED", False):
            v = _make_validator(cameras={"cam1": "rtsp://x:554/s"})
            f = v.validate_channel("cam1", "rtsp://x:554/s", _one_task())
        assert f is None


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: stream reachable
# ─────────────────────────────────────────────────────────────────────────────

class TestStage2StreamReachable:
    def test_fails_when_probe_returns_false(self):
        with patch("services.stream_validator._PROBE_ENABLED", True), \
             patch("services.stream_validator._probe_stream_url", return_value=(False, "timeout")):
            v = _make_validator()
            f = v.validate_channel("cam1", "rtsp://x:554/s", _one_task())
        assert f is not None
        assert f.stage == 2
        assert f.error_code == STREAM_UNREACHABLE

    def test_passes_when_probe_returns_true(self):
        with patch("services.stream_validator._PROBE_ENABLED", True), \
             patch("services.stream_validator._probe_stream_url", return_value=(True, "ok")):
            v = _make_validator()
            f = v.validate_channel("cam1", "rtsp://x:554/s", _one_task())
        assert f is None

    def test_skipped_when_probe_disabled(self):
        with patch("services.stream_validator._PROBE_ENABLED", False):
            v = _make_validator()
            f = v.validate_channel("cam1", "rtsp://x:554/s", _one_task())
        assert f is None

    def test_probe_local_file_exists(self, tmp_path):
        fpath = tmp_path / "test.mp4"
        fpath.write_bytes(b"fake")
        reachable, detail = _probe_stream_url(str(fpath), 2.0)
        assert reachable is True
        assert "exists" in detail

    def test_probe_local_file_missing(self, tmp_path):
        reachable, detail = _probe_stream_url(str(tmp_path / "nope.mp4"), 2.0)
        assert reachable is False

    def test_probe_rtsp_connect_refused(self):
        # Bind a port and immediately close it so the connection is refused.
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.bind(("127.0.0.1", 0))
        port = srv.getsockname()[1]
        srv.close()
        reachable, detail = _probe_stream_url(f"rtsp://127.0.0.1:{port}/stream", 1.0)
        assert reachable is False

    def test_probe_rtsp_success(self):
        # Start a real listening socket to simulate an RTSP server accepting connections.
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port = srv.getsockname()[1]
        try:
            reachable, detail = _probe_stream_url(f"rtsp://127.0.0.1:{port}/stream", 2.0)
            assert reachable is True
        finally:
            srv.close()


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: tasks assigned
# ─────────────────────────────────────────────────────────────────────────────

class TestStage3TasksAssigned:
    def test_fails_when_no_tasks(self):
        with patch("services.stream_validator._PROBE_ENABLED", False):
            v = _make_validator(tasks=[_TASK_BASE])
            f = v.validate_channel("cam1", "rtsp://x:554/s", [])
        assert f is not None
        assert f.stage == 3
        assert f.error_code == NO_TASK_ASSIGNED

    def test_passes_with_one_task(self):
        with patch("services.stream_validator._PROBE_ENABLED", False):
            v = _make_validator()
            f = v.validate_channel("cam1", "rtsp://x:554/s", _one_task())
        assert f is None


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4: task exists in registry
# ─────────────────────────────────────────────────────────────────────────────

class TestStage4TaskExists:
    def test_fails_when_task_missing_from_registry(self):
        reg = _make_task_registry([])  # empty registry
        cams = {"cam1": "rtsp://x:554/s"}
        v = StreamValidator(_make_cam_registry(cams), reg)
        with patch("services.stream_validator._PROBE_ENABLED", False):
            f = v.validate_channel("cam1", "rtsp://x:554/s", _one_task(taskId=999))
        assert f is not None
        assert f.stage == 4
        assert f.error_code == TASK_NOT_FOUND

    def test_passes_when_task_in_registry(self):
        with patch("services.stream_validator._PROBE_ENABLED", False):
            v = _make_validator()
            f = v.validate_channel("cam1", "rtsp://x:554/s", _one_task())
        assert f is None


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5: task enabled
# ─────────────────────────────────────────────────────────────────────────────

class TestStage5TaskEnabled:
    def test_fails_when_task_disabled(self):
        with patch("services.stream_validator._PROBE_ENABLED", False):
            v = _make_validator()
            f = v.validate_channel("cam1", "rtsp://x:554/s", _one_task(enable=False))
        assert f is not None
        assert f.stage == 5
        assert f.error_code == TASK_DISABLED
        assert "disabled" in f.message.lower()

    def test_passes_when_task_enabled(self):
        with patch("services.stream_validator._PROBE_ENABLED", False):
            v = _make_validator()
            f = v.validate_channel("cam1", "rtsp://x:554/s", _one_task(enable=True))
        assert f is None


# ─────────────────────────────────────────────────────────────────────────────
# Stage 6: camera-task mapping
# ─────────────────────────────────────────────────────────────────────────────

class TestStage6CameraTaskMapping:
    def test_fails_when_channel_id_mismatch(self):
        with patch("services.stream_validator._PROBE_ENABLED", False):
            v = _make_validator(cameras={"cam1": "rtsp://x:554/s"})
            # Task says channelId=cam2, but we are starting cam1
            f = v.validate_channel("cam1", "rtsp://x:554/s", _one_task(channelId="cam2"))
        assert f is not None
        assert f.stage == 6
        assert f.error_code == CAMERA_TASK_MISMATCH

    def test_passes_when_channel_id_matches(self):
        with patch("services.stream_validator._PROBE_ENABLED", False):
            v = _make_validator()
            f = v.validate_channel("cam1", "rtsp://x:554/s", _one_task(channelId="cam1"))
        assert f is None


# ─────────────────────────────────────────────────────────────────────────────
# Stage 7: task config valid
# ─────────────────────────────────────────────────────────────────────────────

class TestStage7TaskConfigValid:
    def test_fails_with_empty_area_position(self):
        with patch("services.stream_validator._PROBE_ENABLED", False):
            v = _make_validator()
            f = v.validate_channel(
                "cam1", "rtsp://x:554/s",
                _one_task(areaPosition="[]"),
            )
        assert f is not None
        assert f.stage == 7
        assert f.error_code == TASK_CONFIG_INVALID

    def test_fails_with_invalid_json_area_position(self):
        with patch("services.stream_validator._PROBE_ENABLED", False):
            v = _make_validator()
            f = v.validate_channel(
                "cam1", "rtsp://x:554/s",
                _one_task(areaPosition="{not valid json}"),
            )
        assert f is not None
        assert f.stage == 7
        assert f.error_code == TASK_CONFIG_INVALID

    def test_fails_with_no_valid_points(self):
        bad_area = json.dumps([{"line_id": "1", "point": [{"x": 0}]}])
        with patch("services.stream_validator._PROBE_ENABLED", False):
            v = _make_validator()
            f = v.validate_channel("cam1", "rtsp://x:554/s", _one_task(areaPosition=bad_area))
        assert f is not None
        assert f.stage == 7

    def test_fails_with_out_of_range_threshold(self):
        error = _validate_task_config({**_TASK_BASE, "threshold": 150})
        assert error is not None
        assert "threshold" in error

    def test_passes_with_valid_config(self):
        with patch("services.stream_validator._PROBE_ENABLED", False):
            v = _make_validator()
            f = v.validate_channel("cam1", "rtsp://x:554/s", _one_task())
        assert f is None


# ─────────────────────────────────────────────────────────────────────────────
# Strict order: first failure short-circuits
# ─────────────────────────────────────────────────────────────────────────────

class TestStrictOrder:
    def test_stage1_blocks_before_stage2(self):
        """Stage 1 failure should be returned without reaching the probe."""
        probe_called = []
        with patch("services.stream_validator._probe_stream_url",
                   side_effect=lambda *a, **kw: probe_called.append(True) or (True, "ok")):
            v = _make_validator(cameras={})  # cam1 not registered
            f = v.validate_channel("cam1", "rtsp://x:554/s", _one_task())
        assert f is not None
        assert f.stage == 1
        assert not probe_called, "Probe must not be called when stage 1 fails"

    def test_all_pass_returns_none(self):
        with patch("services.stream_validator._PROBE_ENABLED", False):
            v = _make_validator()
            f = v.validate_channel("cam1", "rtsp://x:554/s", _one_task())
        assert f is None

    def test_validation_failure_has_required_fields(self):
        v = _make_validator(cameras={})
        f = v.validate_channel("cam1", "rtsp://x:554/s", _one_task())
        assert f is not None
        d = f.to_dict()
        for key in ("stage", "stage_name", "error_code", "message", "stream_id", "timestamp", "timestamp_utc"):
            assert key in d, f"Missing key: {key}"

    def test_user_message_format(self):
        v = _make_validator(cameras={})
        f = v.validate_channel("cam1", "rtsp://x:554/s", _one_task())
        assert f is not None
        msg = f.user_message()
        assert "Annotation blocked" in msg
        assert f.error_code in msg
        assert str(f.stage) in msg


# ─────────────────────────────────────────────────────────────────────────────
# _validate_task_config standalone
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateTaskConfig:
    def test_valid_cross_line(self):
        assert _validate_task_config(_TASK_BASE) is None

    def test_missing_algorithm_type(self):
        assert _validate_task_config({**_TASK_BASE, "algorithmType": ""}) is not None

    def test_invalid_threshold_string(self):
        assert _validate_task_config({**_TASK_BASE, "threshold": "bad"}) is not None

    def test_threshold_zero_is_valid(self):
        assert _validate_task_config({**_TASK_BASE, "threshold": 0}) is None

    def test_threshold_100_is_valid(self):
        assert _validate_task_config({**_TASK_BASE, "threshold": 100}) is None

    def test_cross_line_empty_area_position(self):
        error = _validate_task_config({**_TASK_BASE, "areaPosition": ""})
        assert error is not None

    def test_non_cross_line_no_area_check(self):
        task = {**_TASK_BASE, "algorithmType": "PHONE_USAGE", "areaPosition": ""}
        assert _validate_task_config(task) is None
