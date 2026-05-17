"""
tests/unit/test_task_worker_runtime.py
---------------------------------------
Unit tests for runtime task invalidation in ``task_worker.run_task_worker``.

Covers:
  - Worker sets worker_ready_event after successful task init.
  - Worker exits cleanly when task is marked disabled in task_validity_map.
  - Worker exits cleanly when task is removed from task_validity_map.
  - Worker emits a structured PIPELINE_ERROR SSE event on invalidation.
  - Worker emits a WORKER_ERROR event on frame processing exceptions.
  - Worker records init failure in shared_state when algorithm init fails.

No real Redis, RTSP, or YOLO models are needed.
"""

from __future__ import annotations

import json
import multiprocessing
import queue
import threading
import time
from typing import Any, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from task_worker import run_task_worker, _check_task_validity, _record_init_failure
from utils.error_codes import (
    TASK_REMOVED_RUNTIME,
    TASK_DISABLED_RUNTIME,
    WORKER_INIT_FAILED,
    WORKER_ERROR,
)


# ─────────────────────────────────────────────────────────────────────────────
# Minimal in-process helpers (no multiprocessing.Manager)
# ─────────────────────────────────────────────────────────────────────────────

class _SimpleEvent:
    """A threading.Event that also has the multiprocessing.Event interface."""
    def __init__(self):
        self._ev = threading.Event()

    def set(self):
        self._ev.set()

    def is_set(self) -> bool:
        return self._ev.is_set()

    def wait(self, timeout=None) -> bool:
        return self._ev.wait(timeout=timeout)

    def clear(self):
        self._ev.clear()


class _SimpleDict(dict):
    """A plain dict with the same interface as multiprocessing.Manager().dict()."""
    pass


def _run_worker_in_thread(
    task_config: dict,
    task_queue,
    result_queue,
    stop_event,
    *,
    task_validity_map=None,
    worker_ready_event=None,
    shared_state=None,
    timeout: float = 3.0,
) -> threading.Thread:
    """Spawn run_task_worker in a thread (avoids full multiprocessing overhead)."""
    t = threading.Thread(
        target=run_task_worker,
        args=("cam1", task_config, task_queue, result_queue, stop_event),
        kwargs={
            "worker_ready_event": worker_ready_event,
            "task_validity_map": task_validity_map,
            "shared_state": shared_state,
        },
        daemon=True,
    )
    t.start()
    return t


_BASE_TASK = {
    "taskId": 1,
    "taskName": "test",
    "algorithmType": "CROSS_LINE",
    "channelId": "cam1",
    "enable": True,
    "threshold": 50,
    "areaPosition": json.dumps(
        [{"line_id": "1", "point": [{"x": 0, "y": 0}, {"x": 100, "y": 0}], "direction": 0}]
    ),
}


def _mock_task_registry(algorithm: str = "CROSS_LINE"):
    """Return a fake TASK_REGISTRY that instantiates a no-op callable."""
    class _NoOpTask:
        def __init__(self, cfg):
            pass
        def __call__(self, payload):
            return []

    return {algorithm: _NoOpTask}


# task_worker does `from services import TASK_REGISTRY` inside run_task_worker,
# so we must patch it on the `services` module, not on `task_worker`.
_PATCH_REGISTRY = "services.TASK_REGISTRY"


def _make_stop_event() -> _SimpleEvent:
    return _SimpleEvent()


def _make_validity_map(**kwargs) -> _SimpleDict:
    return _SimpleDict(kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# worker_ready_event
# ─────────────────────────────────────────────────────────────────────────────

class TestWorkerReadyEvent:
    def test_ready_event_set_after_successful_init(self):
        ready = _SimpleEvent()
        stop = _make_stop_event()
        task_queue = queue.Queue()
        result_queue = queue.Queue()

        with patch(_PATCH_REGISTRY, _mock_task_registry()):
            t = _run_worker_in_thread(
                _BASE_TASK, task_queue, result_queue, stop,
                worker_ready_event=ready,
            )
            fired = ready.wait(timeout=2.0)
            stop.set()
            t.join(timeout=2.0)

        assert fired, "worker_ready_event must be set after task initialisation"

    def test_ready_event_not_set_on_init_failure(self):
        ready = _SimpleEvent()
        stop = _make_stop_event()
        task_queue = queue.Queue()
        result_queue = queue.Queue()
        shared = _SimpleDict({"cam1": {}})

        def _bad_init(cfg):
            raise RuntimeError("model files missing")

        bad_registry = {"CROSS_LINE": _bad_init}

        with patch(_PATCH_REGISTRY, bad_registry):
            t = _run_worker_in_thread(
                _BASE_TASK, task_queue, result_queue, stop,
                worker_ready_event=ready,
                shared_state=shared,
            )
            t.join(timeout=2.0)

        assert not ready.is_set(), "ready event must NOT be set when init fails"
        # init error should be recorded in shared_state
        assert shared.get("cam1", {}).get("worker_init_error") is not None

    def test_worker_exits_immediately_on_init_failure(self):
        stop = _make_stop_event()
        task_queue = queue.Queue()
        result_queue = queue.Queue()

        def _bad_init(cfg):
            raise ValueError("cannot load weights")

        with patch(_PATCH_REGISTRY, {"CROSS_LINE": _bad_init}):
            t = _run_worker_in_thread(_BASE_TASK, task_queue, result_queue, stop)
            t.join(timeout=2.0)

        assert not t.is_alive(), "Worker must exit without hanging when init fails"


# ─────────────────────────────────────────────────────────────────────────────
# Runtime task removal
# ─────────────────────────────────────────────────────────────────────────────

class TestRuntimeTaskRemoval:
    def test_worker_exits_when_task_removed_from_map(self):
        """Task deleted from registry → worker emits event and stops cleanly."""
        stop = _make_stop_event()
        task_queue = queue.Queue()
        result_queue = queue.Queue()
        validity_map = _make_validity_map()  # task_id "1" absent → removed
        ready = _SimpleEvent()

        with patch(_PATCH_REGISTRY, _mock_task_registry()), \
             patch.dict("os.environ", {"TASK_VALIDITY_POLL_SEC": "0.1"}):
            t = _run_worker_in_thread(
                _BASE_TASK, task_queue, result_queue, stop,
                worker_ready_event=ready,
                task_validity_map=validity_map,
            )
            ready.wait(timeout=2.0)
            t.join(timeout=3.0)

        assert not t.is_alive(), "Worker must stop when task is missing from validity map"

        # Check that a PIPELINE_ERROR was emitted
        events = []
        while not result_queue.empty():
            events.append(result_queue.get_nowait())
        error_events = [e for e in events if e.get("eventType") == "PIPELINE_ERROR"]
        assert len(error_events) >= 1, "Must emit a PIPELINE_ERROR when task removed"
        assert error_events[0]["error_code"] == TASK_REMOVED_RUNTIME

    def test_worker_exits_when_task_disabled_in_map(self):
        """Task disabled → worker emits TASK_DISABLED_RUNTIME and stops."""
        stop = _make_stop_event()
        task_queue = queue.Queue()
        result_queue = queue.Queue()
        # Task exists but is disabled
        validity_map = _make_validity_map(**{"1": {"enabled": False, "exists": True}})
        ready = _SimpleEvent()

        with patch(_PATCH_REGISTRY, _mock_task_registry()), \
             patch.dict("os.environ", {"TASK_VALIDITY_POLL_SEC": "0.1"}):
            t = _run_worker_in_thread(
                _BASE_TASK, task_queue, result_queue, stop,
                worker_ready_event=ready,
                task_validity_map=validity_map,
            )
            ready.wait(timeout=2.0)
            t.join(timeout=3.0)

        assert not t.is_alive(), "Worker must stop when task is disabled"

        events = []
        while not result_queue.empty():
            events.append(result_queue.get_nowait())
        error_events = [e for e in events if e.get("eventType") == "PIPELINE_ERROR"]
        assert any(e["error_code"] == TASK_DISABLED_RUNTIME for e in error_events), \
            f"Must emit TASK_DISABLED_RUNTIME. Got: {[e.get('error_code') for e in error_events]}"

    def test_worker_continues_when_task_valid(self):
        """When validity map shows task enabled and exists, worker keeps running."""
        stop = _make_stop_event()
        task_queue = queue.Queue()
        result_queue = queue.Queue()
        validity_map = _make_validity_map(**{"1": {"enabled": True, "exists": True}})
        ready = _SimpleEvent()

        with patch(_PATCH_REGISTRY, _mock_task_registry()), \
             patch.dict("os.environ", {"TASK_VALIDITY_POLL_SEC": "0.1"}):
            t = _run_worker_in_thread(
                _BASE_TASK, task_queue, result_queue, stop,
                worker_ready_event=ready,
                task_validity_map=validity_map,
            )
            ready.wait(timeout=2.0)
            time.sleep(0.5)  # Let a few validity poll cycles pass
            assert t.is_alive(), "Worker must keep running when task is valid"
            stop.set()
            t.join(timeout=2.0)


# ─────────────────────────────────────────────────────────────────────────────
# Frame processing errors → structured WORKER_ERROR events
# ─────────────────────────────────────────────────────────────────────────────

class TestWorkerErrorEvents:
    def test_frame_processing_exception_emits_pipeline_error(self):
        """Task raising an exception must emit WORKER_ERROR and continue running."""
        stop = _make_stop_event()
        task_queue = queue.Queue()
        result_queue = queue.Queue()
        ready = _SimpleEvent()

        class _BoomTask:
            def __init__(self, cfg):
                pass
            def __call__(self, payload):
                raise RuntimeError("model exploded")

        validity_map = _make_validity_map(**{"1": {"enabled": True, "exists": True}})

        with patch(_PATCH_REGISTRY, {"CROSS_LINE": _BoomTask}), \
             patch.dict("os.environ", {"TASK_VALIDITY_POLL_SEC": "60"}):
            t = _run_worker_in_thread(
                _BASE_TASK, task_queue, result_queue, stop,
                worker_ready_event=ready,
                task_validity_map=validity_map,
            )
            ready.wait(timeout=2.0)
            task_queue.put({"camera_id": "cam1", "frame_id": 1,
                            "timestamp": "2026-05-15T00:00:00", "frame_b64": "",
                            "detection": {"items": [], "count": 0}})
            time.sleep(0.5)
            assert t.is_alive(), "Worker must survive frame processing exceptions"
            stop.set()
            t.join(timeout=2.0)

        events = []
        while not result_queue.empty():
            events.append(result_queue.get_nowait())
        error_events = [e for e in events if e.get("eventType") == "PIPELINE_ERROR"]
        assert any(e["error_code"] == WORKER_ERROR for e in error_events), \
            f"Expected WORKER_ERROR event. Got: {error_events}"


# ─────────────────────────────────────────────────────────────────────────────
# _check_task_validity standalone
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckTaskValidity:
    def test_returns_none_when_task_enabled(self):
        m = {"1": {"enabled": True, "exists": True}}
        assert _check_task_validity(m, "1") is None

    def test_returns_removed_when_key_absent(self):
        result = _check_task_validity({}, "1")
        assert result is not None
        code, msg = result
        assert code == TASK_REMOVED_RUNTIME

    def test_returns_removed_when_exists_false(self):
        result = _check_task_validity({"1": {"enabled": True, "exists": False}}, "1")
        assert result is not None
        code, _ = result
        assert code == TASK_REMOVED_RUNTIME

    def test_returns_disabled_when_enabled_false(self):
        result = _check_task_validity({"1": {"enabled": False, "exists": True}}, "1")
        assert result is not None
        code, _ = result
        assert code == TASK_DISABLED_RUNTIME

    def test_none_map_returns_none(self):
        assert _check_task_validity(None, "1") is None


# ─────────────────────────────────────────────────────────────────────────────
# _record_init_failure
# ─────────────────────────────────────────────────────────────────────────────

class TestRecordInitFailure:
    def test_writes_error_to_shared_state(self):
        shared = _SimpleDict({"cam1": {}})
        _record_init_failure(shared, "cam1", "42", "bad thing happened")
        assert "worker_init_error" in shared["cam1"]
        assert "bad thing happened" in shared["cam1"]["worker_init_error"]

    def test_works_with_none_shared_state(self):
        # Should not raise
        _record_init_failure(None, "cam1", "42", "error")

    def test_missing_camera_row_does_not_raise(self):
        shared = _SimpleDict()  # cam1 not in dict
        _record_init_failure(shared, "cam1", "42", "error")
