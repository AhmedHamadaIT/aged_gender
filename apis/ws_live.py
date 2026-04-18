"""
apis/ws_live.py
---------------
WebSocket endpoints for live annotated frame streaming.

Endpoints (registered in app.py):

  WS  /cameras/{camera_id}/live
      Binary stream of annotated JPEG frames for one camera.
      Each WebSocket message is raw JPEG bytes — no base64, no JSON wrapper.

  WS  /cameras/{camera_id}/events
      JSON stream of detection events for one camera (task results).

Transport rules
  - Frames:  Redis channel  live:frame:{camera_id}   → binary WS message
  - Events:  Redis channel  live:event:{camera_id}   → text WS message (JSON)
  - Backpressure: if send() takes > SEND_TIMEOUT_MS the frame is dropped and
    the loop continues — the client is never queued, the server never blocks.
  - Reconnect: WebSocket does not auto-reconnect. The browser must implement
    a reconnect loop (ws.onclose = () => setTimeout(connect, 2000)).

Browser usage (frames):
    const ws = new WebSocket("ws://host/cameras/cam1/live");
    ws.binaryType = "arraybuffer";
    ws.onmessage = e => {
        const blob = new Blob([e.data], { type: "image/jpeg" });
        document.getElementById("stream").src = URL.createObjectURL(blob);
    };
    ws.onclose = () => setTimeout(() => connect("cam1"), 2000);

Browser usage (events):
    const ws = new WebSocket("ws://host/cameras/cam1/events");
    ws.onmessage = e => console.log(JSON.parse(e.data));
    ws.onclose = () => setTimeout(() => connect("cam1"), 2000);
"""

from __future__ import annotations

import asyncio
import os
from typing import Optional

from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

# Backpressure timeout: drop the frame if the client cannot receive within this
# many milliseconds. This prevents TCP buffer bloat on slow or hidden browser tabs.
_SEND_TIMEOUT_S = float(os.getenv("WS_SEND_TIMEOUT_MS", "50")) / 1000.0


async def _get_redis_async():
    """Return a new async Redis client, or None if redis is unavailable."""
    try:
        import redis.asyncio as aioredis
        url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        client = aioredis.from_url(url, socket_connect_timeout=2)
        await client.ping()
        return client
    except Exception as exc:
        print(f"[ws_live] Redis unavailable: {exc}")
        return None


async def live_frames_ws(websocket: WebSocket, camera_id: str) -> None:
    """
    Stream annotated JPEG frames for *camera_id* over a binary WebSocket.

    Each message sent to the client is raw JPEG bytes.
    Frames are dropped (not queued) if the client cannot receive within
    WS_SEND_TIMEOUT_MS (default 50 ms).
    """
    await websocket.accept()

    redis_client = await _get_redis_async()
    if redis_client is None:
        await websocket.close(code=1011, reason="Redis unavailable")
        return

    channel = f"live:frame:{camera_id}"
    try:
        async with redis_client.pubsub() as ps:
            await ps.subscribe(channel)
            async for msg in ps.listen():
                if websocket.client_state != WebSocketState.CONNECTED:
                    break
                if msg["type"] != "message":
                    continue
                jpeg_bytes: bytes = msg["data"]
                try:
                    await asyncio.wait_for(
                        websocket.send_bytes(jpeg_bytes),
                        timeout=_SEND_TIMEOUT_S,
                    )
                except asyncio.TimeoutError:
                    pass  # client is too slow — drop frame, keep going
                except WebSocketDisconnect:
                    break
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        print(f"[ws_live/{camera_id}] Unexpected error: {exc}")
    finally:
        try:
            await redis_client.aclose()
        except Exception:
            pass


async def live_events_ws(websocket: WebSocket, camera_id: str) -> None:
    """
    Stream JSON detection events for *camera_id* over a text WebSocket.

    Each message is a JSON string (same shape as GET /detection/stream SSE events).
    Events are dropped if the client cannot receive within WS_SEND_TIMEOUT_MS.
    """
    await websocket.accept()

    redis_client = await _get_redis_async()
    if redis_client is None:
        await websocket.close(code=1011, reason="Redis unavailable")
        return

    channel = f"live:event:{camera_id}"
    try:
        async with redis_client.pubsub() as ps:
            await ps.subscribe(channel)
            async for msg in ps.listen():
                if websocket.client_state != WebSocketState.CONNECTED:
                    break
                if msg["type"] != "message":
                    continue
                json_str: str = msg["data"].decode("utf-8") if isinstance(msg["data"], bytes) else msg["data"]
                try:
                    await asyncio.wait_for(
                        websocket.send_text(json_str),
                        timeout=_SEND_TIMEOUT_S,
                    )
                except asyncio.TimeoutError:
                    pass  # slow client — drop event
                except WebSocketDisconnect:
                    break
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        print(f"[ws_live/{camera_id}/events] Unexpected error: {exc}")
    finally:
        try:
            await redis_client.aclose()
        except Exception:
            pass
