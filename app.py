"""
app.py
------
Application entry point — owns all routes and startup.

Workflow:
    1. POST /cameras                  → register cameras (id → rtsp_url)
    2. POST /api/tasks                → register tasks (algorithmType, channelId, config)
    3. POST /detection/start          → start processing
    4. GET  /detection/stream         → SSE stream of task events (broadcast, optional filters)
    5. GET  /detection/status         → monitor camera status
    6. POST /detection/stop           → stop processing

Cashier monitor (algorithmType CASHIER_BOX_OPEN on /api/tasks): HTTP routes under
``/cashier/*`` (status, events, zones, SSE streams). Real-time cashier UI uses
those endpoints; crossing events still use GET /detection/stream.

SSE stream events (one per crossing, per task):
{
    "eventId"     : "...",
    "eventType"   : "CROSS_LINE",
    "timestamp"   : 1774310401528,
    "timestampUTC": "2026-04-02T...",
    "taskId"      : 13,
    "taskName"    : "customer_walkin_main",
    "channelId"   : 4,
    "line"        : {"id": "1", "name": "Entrance", "direction": 1},
    "person"      : {"trackingId", "reidFeature", "boundingBox", "attributes", "confidence"},
    "evidence"    : {"captureImage": "...", "sceneImage": "..."}
}

Run with:
    uvicorn app:app --host 0.0.0.0 --port 9000
"""

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Query, Request
from fastapi.responses import StreamingResponse

from apis.cameras   import camera_registry, CameraSetupRequest
from apis.cashier   import router as cashier_router
from apis.detection import detection
from apis.detection_stream import (
    DETECTION_SSE_KEEPALIVE_SEC,
    DetectionSSEBridge,
    StreamFilters,
)
from apis.tasks     import task_registry, TaskConfig
from schemas        import DetectionRequest, DetectionStatus


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start one SSE bridge per process; stop on shutdown."""
    bridge = DetectionSSEBridge(detection.result_queue())
    await bridge.start()
    app.state.detection_sse_bridge = bridge
    try:
        yield
    finally:
        await bridge.stop()
        app.state.detection_sse_bridge = None


app = FastAPI(
    title="Vision Pipeline API",
    version="2.0.0",
    lifespan=lifespan,
)
app.include_router(cashier_router, prefix="/cashier", tags=["Cashier Monitor"])


# ─────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────
@app.get("/")
def root():
    return {"service": "Vision Pipeline API", "version": "2.0.0"}


# ─────────────────────────────────────────────
# Camera routes
# ─────────────────────────────────────────────
@app.post("/cameras")
def camera_add(req: CameraSetupRequest):
    return camera_registry.on_post(req)


@app.get("/cameras")
def camera_list():
    return camera_registry.on_get()


@app.delete("/cameras/{cam_id}")
def camera_delete(cam_id: str):
    return camera_registry.on_delete(cam_id)


# ─────────────────────────────────────────────
# Task routes
# ─────────────────────────────────────────────
@app.post("/api/tasks")
def task_create(config: TaskConfig):
    return task_registry.on_post(config)


@app.get("/api/tasks")
def task_list():
    return task_registry.on_get_all()


@app.get("/api/tasks/{task_id}")
def task_get(task_id: int):
    return task_registry.on_get_one(task_id)


@app.put("/api/tasks/{task_id}")
def task_update(task_id: int, config: TaskConfig):
    return task_registry.on_put(task_id, config)


@app.delete("/api/tasks/{task_id}")
def task_delete(task_id: int):
    return task_registry.on_delete(task_id)


# ─────────────────────────────────────────────
# Detection routes
# ─────────────────────────────────────────────
@app.post("/detection/start")
def detection_start(camera_id: str = None):
    return detection.on_post(DetectionRequest(action="start", camera_id=camera_id))


@app.post("/detection/stop")
def detection_stop(camera_id: str = None):
    action = "stop_all" if not camera_id else "stop"
    return detection.on_post(DetectionRequest(action=action, camera_id=camera_id))


@app.get("/detection/status", response_model=DetectionStatus)
def detection_status():
    return detection.on_get()


@app.get("/detection/stream")
async def detection_stream(
    request: Request,
    taskId: Optional[int] = Query(
        None,
        description="If set, only events for this task id (AND with other filters).",
    ),
    taskName: Optional[str] = Query(
        None,
        description="If set, only events whose task name matches (AND). Not unique across tasks.",
    ),
    eventType: Optional[str] = Query(
        None,
        description="If set, only events with this eventType (e.g. CROSS_LINE).",
    ),
    channelId: Optional[int] = Query(
        None,
        description="If set, only events from this camera channel id.",
    ),
):
    """
    SSE stream — one JSON object per task event across all cameras.

    Multiple clients each receive a copy of every event (in-process broadcast).
    Optional query params filter server-side with AND semantics.

    Idle connections receive ``: ping`` keepalive comments about every 30 seconds.

    Run a single uvicorn worker for one shared broadcast; multiple workers need
    an external message broker.
    """
    bridge: DetectionSSEBridge = request.app.state.detection_sse_bridge
    filters = StreamFilters(
        task_id=taskId,
        task_name=taskName,
        event_type=eventType,
        channel_id=channelId,
    )
    client_q = bridge.subscribe()
    keepalive_sec = DETECTION_SSE_KEEPALIVE_SEC

    def _task_lookup(tid: int):
        return task_registry.get(tid)

    async def event_generator():
        try:
            while True:
                try:
                    event = await asyncio.wait_for(client_q.get(), timeout=keepalive_sec)
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
                    continue
                if not isinstance(event, dict):
                    continue
                if not filters.matches(event, task_lookup=_task_lookup):
                    continue
                yield f"data: {json.dumps(event)}\n\n"
        finally:
            bridge.unsubscribe(client_q)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control"              : "no-cache",
            "X-Accel-Buffering"          : "no",
            "Access-Control-Allow-Origin": "*",
        },
    )
