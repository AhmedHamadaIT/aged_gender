"""Longer scenarios; require INTEGRATION_STACK=1 and docker-compose.test.yml."""

from __future__ import annotations

import pytest

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.usefixtures("live_stack"),
]
