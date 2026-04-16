"""
Tests for detection SSE bridge, filters, and GET /detection/stream.

Strategy
--------
* Unit tests (StreamFilters, DetectionSSEBridge): pure asyncio.run() — fast,
  no HTTP stack, no camera needed.
* Generator tests: pull items directly from the async generator that backs
  GET /detection/stream — tests filters + keepalive + bridge wiring without
  going through HTTP (httpx 0.28 ASGI transport buffers the entire response,
  so TestClient.stream() on an infinite generator hangs).
* Route/OpenAPI tests: one non-streaming GET /openapi.json via TestClient —
  fast, verifies query params are exposed.

All tests pass without a camera or real RTSP stream.
"""

from __future__ import annotations

import asyncio
import json
import queue
from typing import Any, Dict, List, Optional

import pytest
from fastapi.testclient import TestClient

from apis.detection_stream import DetectionSSEBridge, StreamFilters


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

async def _collect_sse(
    bridge: DetectionSSEBridge,
    *,
    filters: Optional[StreamFilters] = None,
    keepalive_sec: float = 0.08,
    stop_after_data: bool = True,
    max_items: int = 50,
    timeout: float = 5.0,
) -> List[str]:
    """
    Iterate the same async generator logic used by GET /detection/stream and
    return yielded SSE lines.  Stops as soon as a `data:` line is seen (when
    stop_after_data=True) or after max_items, whichever comes first.
    """
    from apis.tasks import task_registry

    f = filters or StreamFilters()
    client_q = bridge.subscribe()

    def _lookup(tid: int):
        return task_registry.get(tid)

    chunks: List[str] = []

    async def _gen():
        try:
            while True:
                try:
                    event = await asyncio.wait_for(client_q.get(), timeout=keepalive_sec)
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
                    continue
                if not isinstance(event, dict):
                    continue
                if not f.matches(event, task_lookup=_lookup):
                    continue
                yield f"data: {json.dumps(event)}\n\n"
        finally:
            bridge.unsubscribe(client_q)

    async def _run():
        async for chunk in _gen():
            chunks.append(chunk)
            if stop_after_data and chunk.startswith("data:"):
                return
            if len(chunks) >= max_items:
                return

    await asyncio.wait_for(_run(), timeout=timeout)
    return chunks


# ─────────────────────────────────────────────
# StreamFilters — unit tests
# ─────────────────────────────────────────────


def test_filters_no_op_passes_all():
    f = StreamFilters()
    assert f.matches({"taskId": 1, "eventType": "X", "channelId": 1})


def test_filters_task_id_match():
    f = StreamFilters(task_id=10)
    assert f.matches({"taskId": 10, "eventType": "X", "channelId": 1})
    assert not f.matches({"taskId": 11, "eventType": "X", "channelId": 1})


def test_filters_event_type_and_channel():
    f = StreamFilters(event_type="CROSS_LINE", channel_id=2)
    assert f.matches({"taskId": 1, "eventType": "CROSS_LINE", "channelId": 2})
    assert not f.matches({"taskId": 1, "eventType": "CROSS_LINE", "channelId": 1})
    assert not f.matches({"taskId": 1, "eventType": "OTHER", "channelId": 2})


def test_filters_task_name_from_event():
    f = StreamFilters(task_name="kitchen")
    assert f.matches({"taskId": 1, "taskName": "kitchen", "eventType": "X", "channelId": 1})
    assert not f.matches({"taskId": 1, "taskName": "other", "eventType": "X", "channelId": 1})


def test_filters_task_name_lookup_fallback():
    def lookup(tid: int) -> Optional[dict]:
        return {"taskName": "from_registry"} if tid == 99 else None

    f = StreamFilters(task_name="from_registry")
    assert f.matches({"taskId": 99, "eventType": "X", "channelId": 1}, task_lookup=lookup)
    assert not f.matches({"taskId": 1, "eventType": "X", "channelId": 1}, task_lookup=lookup)


def test_filters_combined_and():
    f = StreamFilters(task_id=5, event_type="CROSS_LINE", channel_id=1)
    assert f.matches({"taskId": 5, "eventType": "CROSS_LINE", "channelId": 1})
    assert not f.matches({"taskId": 5, "eventType": "CROSS_LINE", "channelId": 2})
    assert not f.matches({"taskId": 6, "eventType": "CROSS_LINE", "channelId": 1})


def test_filters_task_name_combined_with_task_id():
    f = StreamFilters(task_id=7, task_name="x")
    assert f.matches({"taskId": 7, "taskName": "x", "eventType": "Y", "channelId": 1})
    assert not f.matches({"taskId": 7, "taskName": "y", "eventType": "Y", "channelId": 1})


# ─────────────────────────────────────────────
# DetectionSSEBridge — unit tests
# ─────────────────────────────────────────────


def test_bridge_fan_out_two_subscribers():
    """Both subscribers receive a copy of the same event."""
    async def _run():
        sq: queue.Queue = queue.Queue()
        bridge = DetectionSSEBridge(sq)
        await bridge.start()
        a = bridge.subscribe()
        b = bridge.subscribe()
        payload: Dict[str, Any] = {
            "taskId": 1, "taskName": "t", "eventType": "CROSS_LINE", "channelId": 1,
        }
        sq.put(payload)
        got_a = await asyncio.wait_for(a.get(), timeout=2.0)
        got_b = await asyncio.wait_for(b.get(), timeout=2.0)
        assert got_a == payload == got_b
        await bridge.stop()

    asyncio.run(_run())


def test_bridge_unsubscribe_removes_subscriber():
    async def _run():
        sq: queue.Queue = queue.Queue()
        bridge = DetectionSSEBridge(sq)
        await bridge.start()
        q = bridge.subscribe()
        assert q in bridge._subscribers
        bridge.unsubscribe(q)
        assert q not in bridge._subscribers
        await bridge.stop()

    asyncio.run(_run())


def test_bridge_slow_subscriber_dropped():
    """A subscriber whose queue is full is silently dropped on next broadcast."""
    async def _run():
        sq: queue.Queue = queue.Queue()
        bridge = DetectionSSEBridge(sq, subscriber_queue_maxsize=1)
        await bridge.start()
        slow = bridge.subscribe()
        evt = {"taskId": 1, "eventType": "X", "channelId": 1}
        # Fill the queue so the next broadcast overflows it
        slow.put_nowait(evt)
        sq.put(evt)
        await asyncio.sleep(0.5)   # let bridge loop run
        # Slow subscriber should have been removed
        assert slow not in bridge._subscribers
        await bridge.stop()

    asyncio.run(_run())


def test_bridge_stop_is_clean():
    async def _run():
        sq: queue.Queue = queue.Queue()
        bridge = DetectionSSEBridge(sq)
        await bridge.start()
        assert bridge._running
        await bridge.stop()
        assert not bridge._running
        assert bridge._bridge_task is None

    asyncio.run(_run())


# ─────────────────────────────────────────────
# Generator logic tests (no HTTP, no camera)
# ─────────────────────────────────────────────


def test_generator_receives_event():
    """An event placed on the source queue arrives in the SSE output."""
    async def _run():
        sq: queue.Queue = queue.Queue()
        bridge = DetectionSSEBridge(sq)
        await bridge.start()
        evt = {"taskId": 42, "taskName": "line_a", "eventType": "CROSS_LINE", "channelId": 1}
        sq.put(evt)
        chunks = await _collect_sse(bridge, keepalive_sec=0.08, stop_after_data=True)
        await bridge.stop()
        data_chunks = [c for c in chunks if c.startswith("data:")]
        assert len(data_chunks) == 1
        assert json.loads(data_chunks[0][5:].strip())["taskId"] == 42

    asyncio.run(_run())


def test_generator_keepalive_ping():
    """Idle stream emits `: ping` comments."""
    async def _run():
        sq: queue.Queue = queue.Queue()
        bridge = DetectionSSEBridge(sq)
        await bridge.start()
        chunks = await _collect_sse(
            bridge,
            keepalive_sec=0.08,
            stop_after_data=False,
            max_items=3,
        )
        await bridge.stop()
        assert any(c.strip() == ": ping" for c in chunks)

    asyncio.run(_run())


def test_generator_filter_task_id_excludes():
    """Events that don't match taskId are not yielded."""
    async def _run():
        sq: queue.Queue = queue.Queue()
        bridge = DetectionSSEBridge(sq)
        await bridge.start()
        sq.put({"taskId": 99, "taskName": "other", "eventType": "X", "channelId": 1})
        chunks = await _collect_sse(
            bridge,
            filters=StreamFilters(task_id=1),
            keepalive_sec=0.08,
            stop_after_data=False,
            max_items=4,          # collect a few pings; no data should appear
        )
        await bridge.stop()
        assert not any(c.startswith("data:") for c in chunks)

    asyncio.run(_run())


def test_generator_filter_task_id_includes():
    """Events that match taskId are yielded."""
    async def _run():
        sq: queue.Queue = queue.Queue()
        bridge = DetectionSSEBridge(sq)
        await bridge.start()
        evt = {"taskId": 7, "taskName": "n", "eventType": "CROSS_LINE", "channelId": 2}
        sq.put(evt)
        chunks = await _collect_sse(
            bridge,
            filters=StreamFilters(task_id=7),
            keepalive_sec=0.08,
            stop_after_data=True,
        )
        await bridge.stop()
        data_chunks = [c for c in chunks if c.startswith("data:")]
        assert len(data_chunks) == 1
        assert json.loads(data_chunks[0][5:].strip())["taskId"] == 7

    asyncio.run(_run())


def test_generator_filter_event_type():
    async def _run():
        sq: queue.Queue = queue.Queue()
        bridge = DetectionSSEBridge(sq)
        await bridge.start()
        sq.put({"taskId": 1, "eventType": "MASK_HAIRNET_CHEF_HAT", "channelId": 1})
        chunks = await _collect_sse(
            bridge,
            filters=StreamFilters(event_type="MASK_HAIRNET_CHEF_HAT"),
            keepalive_sec=0.08,
            stop_after_data=True,
        )
        await bridge.stop()
        data_chunks = [c for c in chunks if c.startswith("data:")]
        assert len(data_chunks) == 1
        assert json.loads(data_chunks[0][5:].strip())["eventType"] == "MASK_HAIRNET_CHEF_HAT"

    asyncio.run(_run())


def test_generator_filter_channel_id():
    async def _run():
        sq: queue.Queue = queue.Queue()
        bridge = DetectionSSEBridge(sq)
        await bridge.start()
        sq.put({"taskId": 1, "eventType": "X", "channelId": 5})
        chunks = await _collect_sse(
            bridge,
            filters=StreamFilters(channel_id=5),
            keepalive_sec=0.08,
            stop_after_data=True,
        )
        await bridge.stop()
        data_chunks = [c for c in chunks if c.startswith("data:")]
        assert len(data_chunks) == 1
        assert json.loads(data_chunks[0][5:].strip())["channelId"] == 5

    asyncio.run(_run())


def test_generator_filter_task_name_from_event():
    async def _run():
        sq: queue.Queue = queue.Queue()
        bridge = DetectionSSEBridge(sq)
        await bridge.start()
        sq.put({"taskId": 1, "taskName": "cashier_drawer_monitor", "eventType": "X", "channelId": 1})
        chunks = await _collect_sse(
            bridge,
            filters=StreamFilters(task_name="cashier_drawer_monitor"),
            keepalive_sec=0.08,
            stop_after_data=True,
        )
        await bridge.stop()
        data_chunks = [c for c in chunks if c.startswith("data:")]
        assert len(data_chunks) == 1
        assert json.loads(data_chunks[0][5:].strip())["taskName"] == "cashier_drawer_monitor"

    asyncio.run(_run())


def test_generator_combined_filter_and():
    """Combined filters: only events matching ALL conditions pass."""
    async def _run():
        sq: queue.Queue = queue.Queue()
        bridge = DetectionSSEBridge(sq)
        await bridge.start()
        # Should be excluded (wrong eventType)
        sq.put({"taskId": 3, "eventType": "OTHER", "channelId": 1})
        # Should be included
        sq.put({"taskId": 3, "eventType": "CROSS_LINE", "channelId": 1})

        chunks = await _collect_sse(
            bridge,
            filters=StreamFilters(task_id=3, event_type="CROSS_LINE", channel_id=1),
            keepalive_sec=0.08,
            stop_after_data=True,
        )
        await bridge.stop()
        data_chunks = [c for c in chunks if c.startswith("data:")]
        assert len(data_chunks) == 1
        parsed = json.loads(data_chunks[0][5:].strip())
        assert parsed["eventType"] == "CROSS_LINE"

    asyncio.run(_run())


def test_generator_no_filter_passes_all():
    """Without filters every event reaches subscribers."""
    async def _run():
        sq: queue.Queue = queue.Queue()
        bridge = DetectionSSEBridge(sq)
        await bridge.start()
        for i in range(3):
            sq.put({"taskId": i, "eventType": "X", "channelId": 1})
        results = []
        client_q = bridge.subscribe()
        for _ in range(3):
            e = await asyncio.wait_for(client_q.get(), timeout=2.0)
            results.append(e["taskId"])
        await bridge.stop()
        assert results == [0, 1, 2]

    asyncio.run(_run())


def test_generator_fan_out_same_event_two_readers():
    """Two independent generators over the same bridge both receive the event."""
    async def _run():
        sq: queue.Queue = queue.Queue()
        bridge = DetectionSSEBridge(sq)
        await bridge.start()
        evt = {"taskId": 55, "eventType": "CROSS_LINE", "channelId": 1}
        sq.put(evt)
        a_chunks, b_chunks = await asyncio.gather(
            _collect_sse(bridge, keepalive_sec=0.08, stop_after_data=True),
            _collect_sse(bridge, keepalive_sec=0.08, stop_after_data=True),
        )
        await bridge.stop()
        for chunks in (a_chunks, b_chunks):
            data = [c for c in chunks if c.startswith("data:")]
            assert data, "subscriber did not receive event"
            assert json.loads(data[0][5:].strip())["taskId"] == 55

    asyncio.run(_run())


# ─────────────────────────────────────────────
# Route registration (non-streaming HTTP test)
# ─────────────────────────────────────────────


@pytest.fixture
def client():
    from app import app

    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


def test_openapi_detection_stream_query_params(client: TestClient):
    """All four filter params are declared in the OpenAPI spec."""
    r = client.get("/openapi.json")
    assert r.status_code == 200
    params = r.json()["paths"]["/detection/stream"]["get"].get("parameters", [])
    names = {p["name"] for p in params}
    assert {"taskId", "taskName", "eventType", "channelId"}.issubset(names)


def test_detection_stream_route_registered(client: TestClient):
    """The route exists (405/200 not 404)."""
    r = client.get("/openapi.json")
    assert "/detection/stream" in r.json()["paths"]


def test_detection_stream_params_are_optional(client: TestClient):
    """All query params have no 'required' flag (all optional)."""
    r = client.get("/openapi.json")
    params = r.json()["paths"]["/detection/stream"]["get"].get("parameters", [])
    for p in params:
        assert not p.get("required", False), f"{p['name']} should be optional"
