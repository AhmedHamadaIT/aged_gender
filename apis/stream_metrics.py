"""
Stream health and quality endpoints.

GET  /stream/metrics          — per-camera stats from detection shared_state
GET  /stream/quality-events   — SSE: Redis stream:quality_events
GET  /stream/live/{camera_id} — SSE: Redis live:frame:{camera_id} as base64 JSON
"""

from __future__ import annotations

import asyncio
import base64 as _b64
import json
import os
import time
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from apis.ws_live import decode_live_frame_message

router = APIRouter(prefix="/stream", tags=["stream"])


def _redis_url() -> str:
    return os.getenv("REDIS_URL", "redis://localhost:6379/0")


@router.get("/metrics")
async def stream_metrics(request: Request):
    """Per-camera pipeline stats.

    Important fields (see ``FrameBus`` / ``stream`` modules):

    - ``fps`` / ``fps_actual`` — measured processing rate.
    - ``drop_rate`` / ``decode_error_rate`` — failed ``VideoCapture.read()`` divided by
      (failed reads + frames received); RTSP/decode health, not task-queue backpressure.
    - ``task_queue_drop_rate`` — fan-out queue could not accept a frame after coalesce retry.
    - ``task_queue_coalesced_by_task`` — oldest queued frame dropped to make room (see ``TASK_QUEUE_COALESCE``).
    - ``reconnects`` — RTSP reconnects after a consecutive-read-failure burst.
    - ``latency_estimate_ms`` — EMA of wall time between consecutive yielded frames (jitter / stalls).
    - ``framebus_process_alive`` / ``last_state_update_age_sec`` — reconciled with the parent
      FrameBus process when available.
    - ``rtsp_backend`` — active ingest: ``gstreamer``, ``adaptive_ffmpeg``, ``opencv_ffmpeg``, etc.
    - ``live_annotation_mode`` — ``opencv`` / ``ultralytics`` / ``none`` for Redis live JPEGs.
    - ``GET /stream/debug/annotation-state`` — annotation + Redis diagnostics per camera.
    """
    detection = getattr(request.app.state, "detection", None)
    if detection is None:
        return []
    shared = detection._shared_state
    out = []
    for cam_id, v in shared.items():
        row = dict(v)
        out.append(detection.enrich_shared_camera_row(str(cam_id), row))
    return out


@router.get("/health")
async def stream_health(request: Request):
    """Per-camera stream health summary, including adaptive FFmpeg probe details."""
    rows = await stream_metrics(request)
    return {"cameras": rows}


@router.get("/resilience-stats")
async def resilience_stats(request: Request):
    """Per-camera resilience counters (circuit state, buffers, respawns, DLQ)."""
    detection = getattr(request.app.state, "detection", None)
    if detection is None:
        return {}
    keys = (
        "reconnects",
        "frames_dropped",
        "events_buffered",
        "events_replayed",
        "respawn_count",
        "redis_circuit_state",
        "task_redis_circuit_state",
        "stream_metrics",
        "stopped_reason",
    )
    out: dict[str, dict] = {}
    for cam_id, v in detection._shared_state.items():
        row = dict(v)
        cid = str(cam_id)
        out[cid] = {k: row.get(k) for k in keys}
    dlq_path = os.getenv("EMBED_DLQ_DB", "./artifacts/embed_dlq.db")
    dlq_size = 0
    try:
        import sqlite3

        conn = sqlite3.connect(dlq_path)
        cur = conn.execute("SELECT COUNT(*) FROM embed_dlq")
        dlq_size = int(cur.fetchone()[0])
        conn.close()
    except Exception:
        pass
    return {"cameras": out, "embed_dlq_pending": dlq_size, "embed_dlq_path": dlq_path}


@router.get("/debug/annotation-state")
async def debug_annotation_state(request: Request):
    """Per-camera FrameBus annotation / live-publish diagnostics (from shared_state)."""
    detection = getattr(request.app.state, "detection", None)
    if detection is None:
        return {"error": "detection service not initialized", "cameras": {}}
    out: dict[str, dict] = {}
    for cam_id, v in detection._shared_state.items():
        row = dict(v)
        cid = str(cam_id)
        out[cid] = {
            "camera_id": cid,
            "live_annotation_mode": row.get("live_annotation_mode"),
            "save_output": row.get("save_output"),
            "redis_connected": row.get("redis_connected"),
            "redis_circuit_state": row.get("redis_circuit_state"),
            "last_live_publish_seq": row.get("last_live_publish_seq"),
            "last_live_frame_had_boxes": row.get("last_live_frame_had_boxes"),
            "live_jpeg_quality": row.get("live_jpeg_quality"),
            "task_queue_jpeg_quality": row.get("task_queue_jpeg_quality"),
            "frame_count": row.get("frame_count"),
            "last_detections": row.get("last_detections"),
            "running": row.get("running"),
            "state_updated_at": row.get("state_updated_at"),
        }
        out[cid] = detection.enrich_shared_camera_row(cid, out[cid])
    return {"cameras": out}


@router.get("/health/{camera_id}")
async def camera_stream_health(camera_id: str, request: Request):
    detection = getattr(request.app.state, "detection", None)
    if detection is None:
        return {"error": "detection service not initialized"}
    row = detection._shared_state.get(camera_id)
    if row is None:
        return {"error": f"Camera {camera_id} not found"}
    return detection.enrich_shared_camera_row(camera_id, dict(row))


@router.get("/quality-events")
async def quality_events_sse(request: Request):
    """SSE — quality ladder change events (JSON on stream:quality_events)."""

    async def _generator():
        try:
            import redis.asyncio as aioredis

            client = aioredis.from_url(
                _redis_url(), socket_connect_timeout=2, decode_responses=True
            )
            try:
                async with client.pubsub() as ps:
                    await ps.subscribe("stream:quality_events")
                    async for msg in ps.listen():
                        if await request.is_disconnected():
                            break
                        if msg["type"] != "message":
                            continue
                        data = msg["data"]
                        if isinstance(data, bytes):
                            data = data.decode("utf-8", errors="replace")
                        yield f"data: {data}\n\n"
            finally:
                await client.aclose()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(
        _generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/live/{camera_id}")
async def live_stream_sse(camera_id: str, request: Request):
    """SSE — annotated JPEG frames as base64 in JSON (from Redis live:frame:{id})."""

    async def _generator():
        try:
            import redis.asyncio as aioredis

            client = aioredis.from_url(
                _redis_url(), socket_connect_timeout=2, decode_responses=False
            )
            try:
                async with client.pubsub() as ps:
                    await ps.subscribe(f"live:frame:{camera_id}")
                    async for msg in ps.listen():
                        if await request.is_disconnected():
                            break
                        if msg["type"] != "message":
                            continue
                        raw_msg: Any = msg["data"]
                        if not isinstance(raw_msg, (bytes, bytearray, str)):
                            continue
                        jpeg_bytes, seq = decode_live_frame_message(raw_msg)
                        payload = json.dumps(
                            {
                                "camera_id": camera_id,
                                "frame_b64": _b64.b64encode(jpeg_bytes).decode("ascii"),
                                "ts": time.time(),
                                "_seq": seq,
                            }
                        )
                        yield f"data: {payload}\n\n"
            finally:
                await client.aclose()
        except asyncio.CancelledError:
            raise

    return StreamingResponse(
        _generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
