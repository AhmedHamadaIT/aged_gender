"""Thread-safe circuit breaker for Redis and other flaky I/O."""

from __future__ import annotations

import logging
import threading
import time
from enum import Enum
from typing import Any, Callable, Optional

log = logging.getLogger(__name__)


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    """Raised when the circuit is open and calls are short-circuited."""


class CircuitBreaker:
    """
    CLOSED: calls pass through; failures increment counter.
    OPEN: after failure_threshold failures, calls fail fast until reset_timeout.
    HALF_OPEN: one trial call; success closes, failure reopens.
    """

    def __init__(
        self,
        *,
        name: str = "default",
        failure_threshold: int = 5,
        reset_timeout_sec: float = 30.0,
        on_state_change: Optional[Callable[[str, CircuitState, CircuitState], None]] = None,
    ) -> None:
        self._name = name
        self._failure_threshold = max(1, int(failure_threshold))
        self._reset_timeout_sec = max(0.1, float(reset_timeout_sec))
        self._on_state_change = on_state_change
        self._lock = threading.Lock()
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._opened_at: Optional[float] = None
        self._last_success_at: Optional[float] = None

    @property
    def state(self) -> CircuitState:
        with self._lock:
            return self._state

    def state_label(self) -> str:
        return self.state.value

    def _transition(self, new: CircuitState) -> None:
        old = self._state
        if old == new:
            return
        self._state = new
        if self._on_state_change:
            try:
                self._on_state_change(self._name, old, new)
            except Exception:  # noqa: BLE001
                log.exception("circuit breaker on_state_change failed name=%s", self._name)
        log.info(
            "resilience_circuit name=%s old=%s new=%s",
            self._name,
            old.value,
            new.value,
            extra={
                "resilience": True,
                "circuit": self._name,
                "from": old.value,
                "to": new.value,
            },
        )

    def allow_request(self) -> bool:
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return True
            if self._state == CircuitState.OPEN:
                assert self._opened_at is not None
                if time.monotonic() - self._opened_at >= self._reset_timeout_sec:
                    self._transition(CircuitState.HALF_OPEN)
                    return True
                return False
            # HALF_OPEN — allow one trial
            return True

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._last_success_at = time.monotonic()
            if self._state != CircuitState.CLOSED:
                self._transition(CircuitState.CLOSED)
            self._opened_at = None

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._state == CircuitState.HALF_OPEN:
                self._opened_at = time.monotonic()
                self._transition(CircuitState.OPEN)
                return
            if self._state == CircuitState.CLOSED and self._failures >= self._failure_threshold:
                self._opened_at = time.monotonic()
                self._transition(CircuitState.OPEN)

    def call(self, fn: Callable[[], Any], default: Any = None) -> Any:
        """Run fn() if circuit allows; on success record_success, on exception record_failure."""
        if not self.allow_request():
            if default is not None:
                return default
            raise CircuitOpenError(self._name)
        try:
            out = fn()
        except Exception:
            self.record_failure()
            raise
        self.record_success()
        return out
