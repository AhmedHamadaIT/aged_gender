"""
Integration tests require a running docker-compose.test.yml stack.

  export INTEGRATION_STACK=1
  export AUTO_COMPOSE=1   # optional: bring stack up for this session
  docker compose -f docker-compose.test.yml up -d --build
  pytest -m integration
"""

from __future__ import annotations

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.usefixtures("live_stack"),
]
