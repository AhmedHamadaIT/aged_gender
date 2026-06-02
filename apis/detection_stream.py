"""
apis/detection_stream.py
------------------------
Detection SSE bridge: fans out detection events to per-client asyncio queues
so multiple SSE subscribers each receive a copy.

Event sources (mutually exclusive from task_worker):
  1. multiprocessing.Queue (result_queue) — when workers have no Redis client
  2. Redis Pub/Sub  live:event:*          — when REDIS_URL works in task processes

task_worker sends each event on only one path so the bridge does not duplicate
delivery to SSE clients.

Used by GET /detection/stream in app.py.
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import queue as queue_std
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

# Default SSE idle keepalive interval (seconds). Override in tests via monkeypatch.
DETECTION_SSE_KEEPALIVE_SEC = float(os.getenv("DETECTION_SSE_KEEPALIVE_SEC", "30"))
SSE_REPLAY_BUFFER = max(10, int(os.getenv("SSE_REPLAY_BUFFER", "200")))
SSE_OVERFLOW_BUFFER = max(1, int(os.getenv("SSE_OVERFLOW_BUFFER", "50")))


@dataclass
class StreamFilters:
    """AND semantics: all set fields must match."""

    task_id: Optional[int] = None
    task_name: Optional[str] = None
    event_type: Optional[str] = None
    channel_id: Optional[str] = None

    def matches(
        self,
        event: Dict[str, Any],
        task_lookup: Optional[Callable[[int], Optional[dict]]] = None,
    ) -> bool:
        if self.task_id is not None and event.get("taskId") != self.task_id:
            return False
        if self.event_type is not None and event.get("eventType") != self.event_type:
            return False
        if self.channel_id is not None:
            ch = event.get("channelId")
            if ch is None or str(ch) != str(self.channel_id):
                return False
        if self.task_name is not None:
            name = event.get("taskName")
            if name is None and task_lookup is not None:
                tid = event.get("taskId")
                if tid is not None:
                    cfg = task_lookup(int(tid))
                    name = cfg.get("taskName") if cfg else None
            if name != self.task_name:
                return False
        return True


class DetectionSSEBridge:
    """
    Reads from two sources and broadcasts each event to all subscriber
    asyncio.Queue instances on the current event loop.

    Source 1 — multiprocessing.Queue (result_queue):
        In-process path. Works with a single uvicorn worker.

    Source 2 — Redis Pub/Sub pattern  live:event:*:
        Multi-worker path. Active only when REDIS_URL is set and Redis
        is reachable. Allows multiple uvicorn workers to serve SSE clients
        from the same Redis broadcast.
    """

    def __init__(
        self,
        source_queue: Any,
        *,
        subscriber_queue_maxsize: int = 100,
    ) -> None:
        self._source = source_queue
        self._subscriber_queue_maxsize = subscriber_queue_maxsize
        self._subscribers: List[asyncio.Queue] = []
        self._overflow: Dict[int, deque] = {}
        self._replay_ring: deque = deque(maxlen=SSE_REPLAY_BUFFER)
        self._seen_seq: deque = deque(maxlen=5000)
        self._running = False
        self._bridge_task: Optional[asyncio.Task] = None
        self._redis_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._bridge_task = asyncio.create_task(self._bridge_loop(), name="detection_sse_bridge")
        self._redis_task  = asyncio.create_task(self._redis_loop(),  name="detection_sse_redis")

    async def stop(self) -> None:
        self._running = False
        for task in (self._bridge_task, self._redis_task):
            if task is not None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._bridge_task = None
        self._redis_task  = None

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=self._subscriber_queue_maxsize)
        self._subscribers.append(q)
        self._overflow[id(q)] = deque(maxlen=SSE_OVERFLOW_BUFFER)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        try:
            self._subscribers.remove(q)
        except ValueError:
            pass
        self._overflow.pop(id(q), None)

    def replay_after(self, after_seq: int) -> List[dict]:
        """Return a copy of buffered events with _seq strictly greater than after_seq."""
        if after_seq <= 0:
            return [copy.deepcopy(e) for e in self._replay_ring]
        return [
            copy.deepcopy(e)
            for e in self._replay_ring
            if int(e.get("_seq") or 0) > after_seq
        ]

    async def replay_after_stream(self, after_seq: int, camera_id: str = "*") -> List[dict]:
        """
        S-7: Read missed events from Redis Streams when REDIS_STREAMS_ENABLED=true.

        Falls back to the in-memory ring if Streams are disabled or Redis is
        unreachable.  The Pub/Sub path remains authoritative; this is read-only
        replay on reconnect.

        ``camera_id`` filters to a specific stream key; use ``"*"`` for all cameras.
        ``after_seq`` is an integer _seq value — events with _seq > after_seq are returned.
        """
        if not os.getenv("REDIS_STREAMS_ENABLED", "false").lower() in ("true", "1", "yes"):
            return self.replay_after(after_seq)

        redis_url = os.getenv("REDIS_URL", "")
        if not redis_url:
            return self.replay_after(after_seq)

        results: List[dict] = []
        try:
            import redis.asyncio as aioredis

            client = aioredis.from_url(redis_url, socket_connect_timeout=2)
            stream_maxlen = int(os.getenv("REDIS_STREAMS_MAXLEN", "500"))

            if camera_id == "*":
                # Scan all known stream keys matching the pattern.
                keys: List[str] = []
                async for k in client.scan_iter("live:events:*", count=100):
                    keys.append(k.decode("utf-8") if isinstance(k, bytes) else k)
            else:
                keys = [f"live:events:{camera_id}"]

            for key in keys:
                try:
                    # XRANGE returns oldest-to-newest; we read the full stream
                    # (bounded by MAXLEN) and filter by seq in Python.
                    entries = await client.xrange(key, count=stream_maxlen)
                    for _entry_id, fields in entries:
                        raw = fields.get(b"data") or fields.get("data")
                        if raw is None:
                            continue
                        try:
                            ev = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
                            if isinstance(ev, dict) and int(ev.get("_seq") or 0) > after_seq:
                                results.append(ev)
                        except Exception:
                            continue
                except Exception:
                    continue

            await client.aclose()
        except Exception:
            # Any Redis error → fall back to in-memory ring.
            return self.replay_after(after_seq)

        if not results:
            # Nothing from Streams — fall back to in-memory ring to avoid empty replay.
            return self.replay_after(after_seq)

        # Sort by _seq ascending so the client receives them in order.
        results.sort(key=lambda e: int(e.get("_seq") or 0))
        return results

    def _record_replay(self, event: dict) -> None:
        seq = int(event.get("_seq") or 0)
        if seq and seq in self._seen_seq:
            return  # de-dupe
        if seq:
            self._seen_seq.append(seq)
        self._replay_ring.append(copy.deepcopy(event))

    def _get_one_blocking(self, timeout: float) -> Optional[Any]:
        """Blocking get with timeout so executor threads are not stuck forever."""
        try:
            return self._source.get(timeout=timeout)
        except queue_std.Empty:
            return None

    async def _drain_overflow_to_main(self, q: asyncio.Queue) -> None:
        ov = self._overflow.get(id(q))
        if not ov:
            return
        while ov:
            try:
                item = ov.popleft()
                q.put_nowait(item)
            except asyncio.QueueFull:
                ov.appendleft(item)
                break
            except Exception:
                break

    async def _bridge_loop(self) -> None:
        loop = asyncio.get_event_loop()
        poll_sec = min(0.5, max(0.05, float(os.getenv("DETECTION_SSE_BRIDGE_POLL_SEC", "0.25"))))
        while self._running:
            try:
                event = await loop.run_in_executor(
                    None,
                    self._get_one_blocking,
                    poll_sec,
                )
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(0.01)
                continue
            if event is None:
                continue
            if not isinstance(event, dict):
                continue
            await self._broadcast(event)

    async def _broadcast(self, event: dict) -> None:
        self._record_replay(event)
        stale: List[asyncio.Queue] = []
        for q in list(self._subscribers):
            await self._drain_overflow_to_main(q)
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                ov = self._overflow.get(id(q))
                if ov is None or len(ov) >= ov.maxlen:
                    stale.append(q)
                else:
                    ov.append(event)
            except Exception:
                stale.append(q)
        for q in stale:
            self.unsubscribe(q)

    async def _redis_loop(self) -> None:
        """Subscribe to Redis live:event:* and broadcast received events."""
        redis_url = os.getenv("REDIS_URL", "")
        if not redis_url:
            return  # Redis not configured — silently skip

        try:
            import redis.asyncio as aioredis
        except ImportError:
            return

        while self._running:
            client = None
            try:
                client = aioredis.from_url(redis_url, socket_connect_timeout=2)
                await client.ping()
                async with client.pubsub() as ps:
                    await ps.psubscribe("live:event:*")
                    async for msg in ps.listen():
                        if not self._running:
                            break
                        if msg["type"] != "pmessage":
                            continue
                        raw = msg["data"]
                        try:
                            event = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
                        except Exception:
                            continue
                        if isinstance(event, dict):
                            await self._broadcast(event)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                if self._running:
                    import logging
                    logging.getLogger(__name__).warning(
                        "detection_sse_redis reconnect after error: %s", exc
                    )
                    await asyncio.sleep(3)  # back off before reconnecting
            finally:
                if client is not None:
                    try:
                        await client.aclose()
                    except Exception:
                        pass
