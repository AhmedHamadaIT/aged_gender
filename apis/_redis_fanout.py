"""
apis/_redis_fanout.py
---------------------
M-3: CameraStreamMultiplexer — one Redis pubsub connection per camera channel,
fanned out to per-client asyncio.Queue objects.

Problem solved: ws_live.py previously created one Redis pubsub subscription
per WebSocket client (N clients × M cameras = N×M connections).  This module
maintains a single Redis subscription per camera and distributes frames to all
connected clients via lightweight in-process asyncio queues.

Usage (in ws_live.py):
    mux = get_camera_mux(camera_id)
    q = await mux.subscribe()
    try:
        while True:
            frame_data = await asyncio.wait_for(q.get(), timeout=30)
            ...
    finally:
        mux.unsubscribe(q)

This is gated behind WS_MUX_ENABLED=true (default false) so the existing
direct-pubsub code in ws_live.py remains the default path.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Dict, Optional

log = logging.getLogger(__name__)

_WS_MUX_ENABLED: bool = os.getenv("WS_MUX_ENABLED", "false").lower() in ("true", "1", "yes")
_MUX_QUEUE_MAXSIZE: int = max(4, int(os.getenv("WS_MUX_QUEUE_MAXSIZE", "16")))
_MUX_REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Global registry: camera_id → CameraStreamMultiplexer
_mux_registry: Dict[str, "CameraStreamMultiplexer"] = {}
_registry_lock: asyncio.Lock = asyncio.Lock()


class CameraStreamMultiplexer:
    """
    Maintains one Redis pubsub subscription for a camera channel and distributes
    arriving frames to all subscribed asyncio.Queue clients.
    """

    def __init__(self, camera_id: str) -> None:
        self.camera_id = camera_id
        self._channel = f"live:frame:{camera_id}"
        self._clients: list[asyncio.Queue] = []
        self._lock = asyncio.Lock()
        self._task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._pubsub_loop(), name=f"mux_{self.camera_id}")

    async def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=_MUX_QUEUE_MAXSIZE)
        async with self._lock:
            self._clients.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        try:
            self._clients.remove(q)
        except ValueError:
            pass

    @property
    def subscriber_count(self) -> int:
        return len(self._clients)

    async def _pubsub_loop(self) -> None:
        import redis.asyncio as aioredis

        while self._running:
            try:
                client = aioredis.from_url(
                    _MUX_REDIS_URL,
                    socket_connect_timeout=2,
                    decode_responses=False,
                )
                async with client.pubsub() as ps:
                    await ps.subscribe(self._channel)
                    async for msg in ps.listen():
                        if not self._running:
                            break
                        if msg["type"] != "message":
                            continue
                        data = msg["data"]
                        async with self._lock:
                            dead = []
                            for q in self._clients:
                                try:
                                    q.put_nowait(data)
                                except asyncio.QueueFull:
                                    # Slow client: drop oldest frame and enqueue fresh one.
                                    try:
                                        q.get_nowait()
                                        q.put_nowait(data)
                                    except Exception:
                                        pass
                                except Exception:
                                    dead.append(q)
                            for dq in dead:
                                try:
                                    self._clients.remove(dq)
                                except ValueError:
                                    pass
            except asyncio.CancelledError:
                break
            except Exception as exc:
                if self._running:
                    log.warning(
                        "CameraStreamMultiplexer[%s] pubsub error: %s — reconnecting",
                        self.camera_id, exc,
                    )
                    await asyncio.sleep(1.0)


async def get_camera_mux(camera_id: str) -> CameraStreamMultiplexer:
    """Return (and lazily start) the singleton multiplexer for a camera."""
    global _mux_registry, _registry_lock
    async with _registry_lock:
        mux = _mux_registry.get(camera_id)
        if mux is None:
            mux = CameraStreamMultiplexer(camera_id)
            await mux.start()
            _mux_registry[camera_id] = mux
            log.info("CameraStreamMultiplexer[%s] started", camera_id)
    return mux


async def release_camera_mux(camera_id: str) -> None:
    """Stop and remove the multiplexer for a camera (call on camera stop)."""
    global _mux_registry, _registry_lock
    async with _registry_lock:
        mux = _mux_registry.pop(camera_id, None)
    if mux is not None:
        await mux.stop()
        log.info("CameraStreamMultiplexer[%s] stopped", camera_id)
