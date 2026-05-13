"""
apis/ws_live.py
---------------
WebSocket endpoints for live annotated frame streaming.

Endpoints (registered in app.py):

  WS  /cameras/{camera_id}/live
      Binary stream of annotated JPEG frames for one camera.
      Each WebSocket message is raw JPEG bytes by default (no base64 wrapper).
      Optional ``WS_INCLUDE_SEQ_HEADER=true``: each message is 4-byte little-endian
      ``uint32`` sequence (Redis ``_seq``) followed by JPEG bytes (for gap detection).

  WS  /cameras/{camera_id}/events
      JSON stream of detection events for one camera (task results).

Transport rules
  - Frames:  Redis channel  live:frame:{camera_id}   → binary WS message
  - Events:  Redis channel  live:event:{camera_id}   → text WS message (JSON)
  - Backpressure: if send() takes > SEND_TIMEOUT_MS the frame is dropped and
    the loop continues — the client is never queued, the server never blocks.
  - Reconnect: WebSocket does not auto-reconnect. The browser should implement
    exponential backoff (see example below).

Browser usage (frames) — memory-safe + rAF-throttled preview:

    const ws = new WebSocket("ws://host/cameras/cam1/live");
    ws.binaryType = "arraybuffer";
    let currentUrl = null;
    let pendingFrame = null;
    let rafScheduled = false;
    function renderLoop() {
      rafScheduled = false;
      if (pendingFrame) {
        if (currentUrl) URL.revokeObjectURL(currentUrl);
        currentUrl = URL.createObjectURL(
          new Blob([pendingFrame], { type: "image/jpeg" })
        );
        document.getElementById("stream").src = currentUrl;
        pendingFrame = null;
      }
    }
    ws.onmessage = (e) => {
      pendingFrame = e.data;
      if (!rafScheduled) {
        rafScheduled = true;
        requestAnimationFrame(renderLoop);
      }
    };
    function connect(cameraId) {
      const ws = new WebSocket(`ws://${location.host}/cameras/${cameraId}/live`);
      ws.binaryType = "arraybuffer";
      let retryDelay = 1000;
      const maxDelay = 30000;
      ws.onopen = () => { retryDelay = 1000; };
      ws.onclose = () => {
        setTimeout(() => connect(cameraId), retryDelay);
        retryDelay = Math.min(retryDelay * 2, maxDelay);
      };
      ws.onerror = () => ws.close();
      ws.onmessage = (e) => { /* attach render logic */ };
      return ws;
    }
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import struct
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
_WS_REDIS_RECONNECT_BASE_MS = max(100, int(os.getenv("WS_REDIS_RECONNECT_DELAY_MS", "1000")))
_WS_REDIS_RECONNECT_MAX_MS = max(
    _WS_REDIS_RECONNECT_BASE_MS, int(os.getenv("WS_REDIS_RECONNECT_MAX_MS", "30000"))
)
_WS_REDIS_MAX_RETRIES = max(1, int(os.getenv("WS_REDIS_MAX_RETRIES", "5")))
_FRAME_RING_MAX = max(10, int(os.getenv("WS_FRAME_REPLAY_BUFFER", os.getenv("SSE_REPLAY_BUFFER", "200"))))

_WS_MAX_FPS = float(os.getenv("WS_MAX_FPS", "15"))
_WS_MIN_FRAME_INTERVAL = 1.0 / _WS_MAX_FPS if _WS_MAX_FPS > 0 else 0.0

_WS_QUALITY_REFRESH_SEC = float(os.getenv("WS_QUALITY_REFRESH_SEC", "1.0"))
_WS_TIER_MAX_FPS = {
    "high": float(os.getenv("WS_TIER_HIGH_FPS", "15")),
    "medium": float(os.getenv("WS_TIER_MEDIUM_FPS", "10")),
    "low": float(os.getenv("WS_TIER_LOW_FPS", "8")),
    "minimal": float(os.getenv("WS_TIER_MINIMAL_FPS", "5")),
}

_WS_BP_REDIS_INTERVAL_SEC = float(os.getenv("WS_BP_REDIS_INTERVAL_SEC", "2.0"))
_ws_bp_last_wall: Dict[str, float] = {}

_WS_INCLUDE_SEQ_HEADER = os.getenv("WS_INCLUDE_SEQ_HEADER", "false").lower() in (
    "true",
    "1",
    "yes",
    "on",
)

# Per-camera ring of (seq, jpeg_bytes) for optional ?last_seq= replay
_frame_ring: Dict[str, deque] = defaultdict(lambda: deque(maxlen=_FRAME_RING_MAX))

# Per-camera rate limit for WebSocket lifecycle warnings (not per-frame).
_last_ws_warn: Dict[str, float] = {}


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


def _sync_redis_get_stream_quality(camera_id: str) -> Optional[str]:
    try:
        import redis as r

        url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        c = r.Redis.from_url(url, socket_connect_timeout=0.3, socket_timeout=0.3)
        v = c.get(f"stream:quality:{camera_id}")
        c.close()
        if v is None:
            return None
        return v.decode() if isinstance(v, bytes) else str(v)
    except Exception:
        return None


def _tier_to_min_frame_interval(tier: Optional[str]) -> float:
    if not tier:
        return _WS_MIN_FRAME_INTERVAL
    key = str(tier).strip().lower()
    cap = _WS_TIER_MAX_FPS.get(key, _WS_MAX_FPS)
    return 1.0 / cap if cap > 0 else 0.0


def _sync_set_ws_backpressure(camera_id: str) -> None:
    try:
        import redis as r

        url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        c = r.Redis.from_url(url, socket_connect_timeout=0.3, socket_timeout=0.3)
        c.setex(f"stream:ws_backpressure:{camera_id}", 8, "1")
        c.close()
    except Exception:
        pass


async def _maybe_publish_ws_backpressure(camera_id: str) -> None:
    now = time.time()
    if now - _ws_bp_last_wall.get(camera_id, 0.0) < _WS_BP_REDIS_INTERVAL_SEC:
        return
    _ws_bp_last_wall[camera_id] = now
    await asyncio.to_thread(_sync_set_ws_backpressure, camera_id)


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


def _ws_redis_backoff_sec(attempt: int) -> float:
    """Exponential backoff with ±20% jitter for Redis reconnect delays."""
    import random

    delay_ms = min(
        _WS_REDIS_RECONNECT_BASE_MS * (2**attempt),
        _WS_REDIS_RECONNECT_MAX_MS,
    )
    jitter = delay_ms * 0.2 * (2.0 * random.random() - 1.0)
    return max(_WS_REDIS_RECONNECT_BASE_MS / 1000.0, (delay_ms + jitter) / 1000.0)


def _rate_limited_ws_warn(camera_id: str, fmt: str, *args: object) -> None:
    now = time.time()
    if now - _last_ws_warn.get(camera_id, 0.0) > 5.0:
        _last_ws_warn[camera_id] = now
        log.warning(fmt, *args)


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


def _frame_ws_payload(jpeg_bytes: bytes, seq: int) -> bytes:
    if _WS_INCLUDE_SEQ_HEADER:
        return struct.pack("<I", seq & 0xFFFFFFFF) + jpeg_bytes
    return jpeg_bytes


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
                    websocket.send_bytes(_frame_ws_payload(jpeg, seq)),
                    timeout=_SEND_TIMEOUT_S,
                )
            except asyncio.TimeoutError:
                break
            except WebSocketDisconnect:
                break
            except RuntimeError:
                break


async def live_frames_ws(websocket: WebSocket, camera_id: str) -> None:
    """
    Stream annotated JPEG frames for *camera_id* over a binary WebSocket.

    Each message sent to the client is raw JPEG bytes (or 4-byte seq + JPEG when
    WS_INCLUDE_SEQ_HEADER=true). Per-connection send rate is capped by WS_MAX_FPS.

    Query param ``last_seq`` (optional): replay buffered frames with sequence
    greater than this value after connect (best-effort ring buffer).
    """
    await websocket.accept()

    last_seq = 0
    q = websocket.query_params.get("last_seq")
    if q and q.isdigit():
        last_seq = int(q)
    await _replay_frames(websocket, camera_id, last_seq)

    last_sent_at = 0.0
    min_interval = _WS_MIN_FRAME_INTERVAL
    last_qos_check = 0.0
    redis_attempt = 0
    while True:
        if websocket.client_state != WebSocketState.CONNECTED:
            break

        redis_client = await _get_redis_async()
        if redis_client is None:
            if redis_attempt >= _WS_REDIS_MAX_RETRIES:
                break
            delay = _ws_redis_backoff_sec(redis_attempt)
            redis_attempt += 1
            await asyncio.sleep(delay)
            continue

        redis_attempt = 0
        channel = f"live:frame:{camera_id}"
        _ws_dead = False
        try:
            async with redis_client.pubsub() as ps:
                await ps.subscribe(channel)
                async for msg in ps.listen():
                    if websocket.client_state != WebSocketState.CONNECTED:
                        _ws_dead = True
                        break
                    if msg["type"] != "message":
                        continue
                    jpeg_bytes, seq = decode_live_frame_message(msg["data"])
                    _record_frame_ring(camera_id, seq, jpeg_bytes)
                    if websocket.client_state != WebSocketState.CONNECTED:
                        _ws_dead = True
                        break
                    nowm = time.monotonic()
                    if nowm - last_qos_check >= _WS_QUALITY_REFRESH_SEC:
                        last_qos_check = nowm
                        tier = await asyncio.to_thread(_sync_redis_get_stream_quality, camera_id)
                        min_interval = _tier_to_min_frame_interval(tier)
                    if min_interval > 0:
                        now = time.monotonic()
                        if now - last_sent_at < min_interval:
                            continue
                        last_sent_at = now
                    try:
                        await asyncio.wait_for(
                            websocket.send_bytes(_frame_ws_payload(jpeg_bytes, seq)),
                            timeout=_SEND_TIMEOUT_S,
                        )
                    except asyncio.TimeoutError:
                        await _maybe_publish_ws_backpressure(camera_id)
                    except WebSocketDisconnect:
                        _ws_dead = True
                        _rate_limited_ws_warn(
                            camera_id,
                            "ws_live/%s frames: WebSocket disconnected during send",
                            camera_id,
                        )
                        break
                    except RuntimeError as exc:
                        _ws_dead = True
                        _rate_limited_ws_warn(
                            camera_id,
                            "ws_live/%s frames: WebSocket send failed (%s)",
                            camera_id,
                            exc,
                        )
                        break
        except WebSocketDisconnect:
            _ws_dead = True
            _rate_limited_ws_warn(
                camera_id,
                "ws_live/%s frames: WebSocket disconnected in pubsub loop",
                camera_id,
            )
        except Exception as exc:
            if websocket.client_state != WebSocketState.CONNECTED:
                _ws_dead = True
                _rate_limited_ws_warn(
                    camera_id,
                    "ws_live/%s frames: Redis loop stopped (client not connected): %s",
                    camera_id,
                    exc,
                )
            else:
                delay = _ws_redis_backoff_sec(redis_attempt)
                log.warning(
                    "ws_live/%s frames redis error: %s — reconnecting in %.1fs (attempt %d/%d)",
                    camera_id,
                    exc,
                    delay,
                    redis_attempt + 1,
                    _WS_REDIS_MAX_RETRIES,
                )
                redis_attempt += 1
                await asyncio.sleep(delay)
        finally:
            try:
                await redis_client.aclose()
            except Exception:
                pass

        if _ws_dead:
            break

    if websocket.client_state == WebSocketState.CONNECTED:
        try:
            await websocket.close(code=1011, reason="Redis unavailable or max retries")
        except Exception:
            pass


async def live_events_ws(websocket: WebSocket, camera_id: str) -> None:
    """
    Stream JSON detection events for *camera_id* over a text WebSocket.

    Each message is a JSON string (same shape as GET /detection/stream SSE events).
    Events are dropped if the client cannot receive within WS_SEND_TIMEOUT_MS.
    """
    await websocket.accept()

    redis_attempt = 0
    while True:
        if websocket.client_state != WebSocketState.CONNECTED:
            break

        redis_client = await _get_redis_async()
        if redis_client is None:
            if redis_attempt >= _WS_REDIS_MAX_RETRIES:
                break
            delay = _ws_redis_backoff_sec(redis_attempt)
            redis_attempt += 1
            await asyncio.sleep(delay)
            continue

        redis_attempt = 0
        channel = f"live:event:{camera_id}"
        _ws_dead = False
        try:
            async with redis_client.pubsub() as ps:
                await ps.subscribe(channel)
                async for msg in ps.listen():
                    if websocket.client_state != WebSocketState.CONNECTED:
                        _ws_dead = True
                        break
                    if msg["type"] != "message":
                        continue
                    json_str: str = msg["data"].decode("utf-8") if isinstance(msg["data"], bytes) else msg["data"]
                    if websocket.client_state != WebSocketState.CONNECTED:
                        _ws_dead = True
                        break
                    try:
                        await asyncio.wait_for(
                            websocket.send_text(json_str),
                            timeout=_SEND_TIMEOUT_S,
                        )
                    except asyncio.TimeoutError:
                        pass
                    except WebSocketDisconnect:
                        _ws_dead = True
                        _rate_limited_ws_warn(
                            camera_id,
                            "ws_live/%s events: WebSocket disconnected during send",
                            camera_id,
                        )
                        break
                    except RuntimeError as exc:
                        _ws_dead = True
                        _rate_limited_ws_warn(
                            camera_id,
                            "ws_live/%s events: WebSocket send failed (%s)",
                            camera_id,
                            exc,
                        )
                        break
        except WebSocketDisconnect:
            _ws_dead = True
            _rate_limited_ws_warn(
                camera_id,
                "ws_live/%s events: WebSocket disconnected in pubsub loop",
                camera_id,
            )
        except Exception as exc:
            if websocket.client_state != WebSocketState.CONNECTED:
                _ws_dead = True
                _rate_limited_ws_warn(
                    camera_id,
                    "ws_live/%s events: Redis loop stopped (client not connected): %s",
                    camera_id,
                    exc,
                )
            else:
                delay = _ws_redis_backoff_sec(redis_attempt)
                log.warning(
                    "ws_live/%s events redis error: %s — reconnecting in %.1fs (attempt %d/%d)",
                    camera_id,
                    exc,
                    delay,
                    redis_attempt + 1,
                    _WS_REDIS_MAX_RETRIES,
                )
                redis_attempt += 1
                await asyncio.sleep(delay)
        finally:
            try:
                await redis_client.aclose()
            except Exception:
                pass

        if _ws_dead:
            break

    if websocket.client_state == WebSocketState.CONNECTED:
        try:
            await websocket.close(code=1011, reason="Redis unavailable or max retries")
        except Exception:
            pass
