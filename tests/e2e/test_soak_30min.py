"""Soak test — run manually before release."""

from __future__ import annotations

import pytest


@pytest.mark.soak
@pytest.mark.skip(reason="30+ minute soak; run explicitly: pytest -m soak tests/e2e/test_soak_30min.py")
def test_soak_placeholder():
    assert False
