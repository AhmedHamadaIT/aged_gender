"""Manual: inject TASK worker sleep via env (see TEST_PLAYBOOK)."""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="Requires TEST_TASK_SLEEP_MS hook in task_worker (not implemented).")
def test_task_queue_backpressure():
    assert False
