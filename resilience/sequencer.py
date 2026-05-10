"""Monotonic sequence IDs shared across processes (Manager.Value + Lock)."""

from __future__ import annotations

from typing import Any


def next_seq(counter: Any, lock: Any) -> int:
    """
    Increment a multiprocessing.Value('Q', 0) under a Manager lock.
    Returns the new sequence value (1-based increment from previous stored value).
    """
    if counter is None or lock is None:
        return 0
    with lock:
        counter.value += 1
        return int(counter.value)


def seq_from_event(event: dict) -> int:
    """Extract sequence from event dict; 0 if missing."""
    try:
        v = event.get("_seq")
        if v is None:
            return 0
        return int(v)
    except (TypeError, ValueError):
        return 0
