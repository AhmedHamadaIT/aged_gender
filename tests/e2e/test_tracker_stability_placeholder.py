"""Deterministic tracker IDs require stable vision fixtures."""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="Use real curated clips + golden event IDs in a future revision.")
def test_tracker_ids_stable_across_loops():
    assert False
