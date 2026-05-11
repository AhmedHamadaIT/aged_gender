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
import base64
import json
import logging
import os
import re
import time
from collections import defaultdict, deque
from typing import Dict, Optional, Tuple

from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

# Valid camera IDs: alphanumeric, underscore, hyphen — no spaces or special chars.
# Reserved JS sentinel values are explicitly rejected.
_RESERVED_CAMERA_IDS = frozenset({"null", "undefined", "none", "nan"})
_VALID_CAMERA_ID_RE = re.compile(r"^[A-Za-z0-9_\-]+$")

log = logging.getLogger(__name__)

# Backpressure timeout: drop the frame if the client cannot receive within this
# many milliseconds. This prevents TCP buffer bloat on slow or hidden browser tabs.
_SEND_TIMEOUT_S = float(os.getenv("WS_SEND_TIMEOUT_MS", "50")) / 1000.0
_WS_REDIS_RECONNECT_DELAY_MS = max(100, int(os.getenv("WS_REDIS_RECONNECT_DELAY_MS", "1000")))
_WS_REDIS_MAX_RETRIES = max(1, int(os.getenv("WS_REDIS_MAX_RETRIES", "5")))
_FRAME_RING_MAX = max(10, int(os.getenv("WS_FRAME_REPLAY_BUFFER", os.getenv("SSE_REPLAY_BUFFER", "200"))))

# Per-camera ring of (seq, jpeg_bytes) for optional ?last_seq= replay
_frame_ring: Dict[str, deque] = defaultdict(lambda: deque(maxlen=_FRAME_RING_MAX))


def validate_camera_id(camera_id: str) -> Optional[str]:
    """
    Validate a camera_id string.

    Returns None if the ID is acceptable, or a short rejection reason string
    if the ID is invalid (so callers can do ``if validate_camera_id(cid): reject``).
    """
    if not camera_id or not camera_id.strip():
        return "empty camera_id"
    if camera_id.lower() in _RESERVED_CAMERA_IDS:
        return f"reserved sentinel value: {camera_id!r}"
    if not _VALID_CAMERA_ID_RE.match(camera_id):
        return f"camera_id contains invalid characters: {camera_id!r}"
    return None


def decode_live_frame_message(data: bytes | str | memoryview) -> Tuple[bytes, int]:
    """
    Redis payload is either legacy raw JPEG bytes or JSON envelope
    {"_seq": int, "jpeg": "<base64>"}.
    """
    if isinstance(data, str):
        raw = data.encode("utf-8")
    elif isinstance(data, memoryview):
        raw = data.tobytes()
    else:
        raw = data
    if len(raw) >= 2 and raw[0:1] == b"{":
        try:
            obj = json.loads(raw.decode("utf-8"))
            jpeg_b64 = obj.get("jpeg", "")
            seq = int(obj.get("_seq", 0))
            return base64.b64decode(jpeg_b64), seq
        except Exception:
            return raw, 0
    return raw, 0


def _record_frame_ring(camera_id: str, seq: int, jpeg: bytes) -> None:
    if seq <= 0:
        _frame_ring[camera_id].append((0, jpeg))
        return
    _frame_ring[camera_id].append((seq, jpeg))


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


async def _replay_frames(
    websocket: WebSocket,
    camera_id: str,
    last_seq: int,
) -> None:
    if last_seq <= 0:
        return
    dq = _frame_ring.get(camera_id)
    if not dq:
        return
    for seq, jpeg in dq:
        if seq > last_seq or seq == 0:
            try:
                await asyncio.wait_for(
                    websocket.send_bytes(jpeg),
                    timeout=_SEND_TIMEOUT_S,
                )
            except (asyncio.TimeoutError, WebSocketDisconnect):
                break


async def live_frames_ws(websocket: WebSocket, camera_id: str) -> None:
    """
    Stream annotated JPEG frames for *camera_id* over a binary WebSocket.

    Each message sent to the client is raw JPEG bytes.
    Frames are dropped (not queued) if the client cannot receive within
    WS_SEND_TIMEOUT_MS (default 50 ms).

    Query param ``last_seq`` (optional): replay buffered frames with sequence
    greater than this value after connect (best-effort ring buffer).
    """
    await websocket.accept()

    last_seq = 0
    q = websocket.query_params.get("last_seq")
    if q and q.isdigit():
        last_seq = int(q)
    await _replay_frames(websocket, camera_id, last_seq)

    redis_attempt = 0
    while redis_attempt < _WS_REDIS_MAX_RETRIES and websocket.client_state == WebSocketState.CONNECTED:
        redis_client = await _get_redis_async()
        if redis_client is None:
            redis_attempt += 1
            await asyncio.sleep(_WS_REDIS_RECONNECT_DELAY_MS / 1000.0)
            continue
        redis_attempt = 0
        channel = f"live:frame:{camera_id}"
        try:
            async with redis_client.pubsub() as ps:
                await ps.subscribe(channel)
                async for msg in ps.listen():
                    if websocket.client_state != WebSocketState.CONNECTED:
                        break
                    if msg["type"] != "message":
                        continue
                    jpeg_bytes, seq = decode_live_frame_message(msg["data"])
                    _record_frame_ring(camera_id, seq, jpeg_bytes)
                    try:
                        await asyncio.wait_for(
                            websocket.send_bytes(jpeg_bytes),
                            timeout=_SEND_TIMEOUT_S,
                        )
                    except asyncio.TimeoutError:
                        pass
                    except WebSocketDisconnect:
                        break
        except WebSocketDisconnect:
            break
        except Exception as exc:
            log.warning("ws_live/%s frames redis loop: %s — reconnecting", camera_id, exc)
            redis_attempt += 1
            await asyncio.sleep(_WS_REDIS_RECONNECT_DELAY_MS / 1000.0)
        finally:
            try:
                await redis_client.aclose()
            except Exception:
                pass

    if websocket.client_state == WebSocketState.CONNECTED:
        await websocket.close(code=1011, reason="Redis unavailable or max retries")


async def live_events_ws(websocket: WebSocket, camera_id: str) -> None:
    """
    Stream JSON detection events for *camera_id* over a text WebSocket.

    Each message is a JSON string (same shape as GET /detection/stream SSE events).
    Events are dropped if the client cannot receive within WS_SEND_TIMEOUT_MS.
    """
    await websocket.accept()

    redis_attempt = 0
    while redis_attempt < _WS_REDIS_MAX_RETRIES and websocket.client_state == WebSocketState.CONNECTED:
        redis_client = await _get_redis_async()
        if redis_client is None:
            redis_attempt += 1
            await asyncio.sleep(_WS_REDIS_RECONNECT_DELAY_MS / 1000.0)
            continue
        redis_attempt = 0
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
                        pass
                    except WebSocketDisconnect:
                        break
        except WebSocketDisconnect:
            break
        except Exception as exc:
            log.warning("ws_live/%s events redis loop: %s — reconnecting", camera_id, exc)
            redis_attempt += 1
            await asyncio.sleep(_WS_REDIS_RECONNECT_DELAY_MS / 1000.0)
        finally:
            try:
                await redis_client.aclose()
            except Exception:
                pass

    if websocket.client_state == WebSocketState.CONNECTED:
        await websocket.close(code=1011, reason="Redis unavailable or max retries")
