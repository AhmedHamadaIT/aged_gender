"""
E2E placeholder: Redis failover / circuit recovery.

Full simulation requires docker-compose with Redis stop/start; this test documents
the intended scenario and asserts resilience helpers import cleanly.
"""

from resilience.circuit_breaker import CircuitBreaker
from resilience.event_buffer import EventBuffer


def test_circuit_breaker_opens_and_half_opens():
    cb = CircuitBreaker(name="test", failure_threshold=2, reset_timeout_sec=0.1)
    assert cb.allow_request()
    cb.record_failure()
    cb.record_failure()
    assert not cb.allow_request()
    import time

    time.sleep(0.15)
    assert cb.allow_request()


def test_event_buffer_memory_then_spill(tmp_path):
    spill = str(tmp_path / "spill.db")
    buf = EventBuffer(max_memory=2, db_path=spill)
    assert buf.append({"a": 1})[0]
    assert buf.append({"a": 2})[0]
    ok, where = buf.append({"a": 3})
    assert ok and where == "disk"
    batch = buf.popleft_batch(10)
    assert len(batch) >= 3
    buf.close()
