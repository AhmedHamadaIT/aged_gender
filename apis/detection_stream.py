"""
apis/detection_stream.py
------------------------
Detection SSE bridge: one reader on the multiprocessing result queue fans out
events to per-client asyncio queues so multiple subscribers each receive a copy.

Used by GET /detection/stream in app.py. Run one uvicorn worker for a single
in-process broadcast; multiple workers need an external broker.
"""

from __future__ import annotations

import asyncio
import os
import queue as queue_std
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

# Default SSE idle keepalive interval (seconds). Override in tests via monkeypatch.
DETECTION_SSE_KEEPALIVE_SEC = float(os.getenv("DETECTION_SSE_KEEPALIVE_SEC", "30"))


@dataclass
class StreamFilters:
    """AND semantics: all set fields must match."""

    task_id: Optional[int] = None
    task_name: Optional[str] = None
    event_type: Optional[str] = None
    channel_id: Optional[int] = None

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
            if ch is None or int(ch) != int(self.channel_id):
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
    Reads blocking from a multiprocessing.Queue and broadcasts each event to
    all subscriber asyncio.Queue instances on the current event loop.
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
        self._running = False
        self._bridge_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._bridge_task = asyncio.create_task(self._bridge_loop(), name="detection_sse_bridge")

    async def stop(self) -> None:
        self._running = False
        if self._bridge_task is not None:
            self._bridge_task.cancel()
            try:
                await self._bridge_task
            except asyncio.CancelledError:
                pass
            self._bridge_task = None

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=self._subscriber_queue_maxsize)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        try:
            self._subscribers.remove(q)
        except ValueError:
            pass

    def _get_one_blocking(self, timeout: float) -> Optional[Any]:
        """Blocking get with timeout so executor threads are not stuck forever."""
        try:
            return self._source.get(timeout=timeout)
        except queue_std.Empty:
            return None

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
        stale: List[asyncio.Queue] = []
        for q in list(self._subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                stale.append(q)
            except Exception:
                stale.append(q)
        for q in stale:
            self.unsubscribe(q)
