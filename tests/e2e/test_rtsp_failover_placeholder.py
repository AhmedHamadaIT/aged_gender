"""Manual: restart MediaMTX mid-run (see docs/TEST_PLAYBOOK.md)."""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="Manual scenario — docker kill mediamtx; automate with testcontainers later.")
def test_rtsp_failover():
    assert False
