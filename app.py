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

from dotenv import load_dotenv

load_dotenv()

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, Query, Request, WebSocket
from fastapi.responses import StreamingResponse

from apis.cameras   import camera_registry, CameraSetupRequest
from apis.cashier   import router as cashier_router
from apis.detection import detection
from apis.stream_metrics import router as stream_metrics_router
from apis.detection_stream import (
    DETECTION_SSE_KEEPALIVE_SEC,
    DetectionSSEBridge,
    StreamFilters,
)
from apis.tasks     import task_registry, TaskConfig
from apis.ws_live   import live_frames_ws, live_events_ws
from schemas        import DetectionRequest, DetectionStatus
from apis.person_search import person_search_api
from apis.semantic_search import semantic_search_api


_API_DESCRIPTION = """
## WebSocket streams

These endpoints use the WebSocket protocol and do not appear as separate operations in the OpenAPI path list.

| URL | Messages |
|-----|----------|
| `WS /cameras/{camera_id}/live` | Binary: one annotated JPEG frame per message. |
| `WS /cameras/{camera_id}/events` | Text: JSON objects with the same shape as `GET /detection/stream` payloads. |
| `WS /tasks/{task_name}/live` | Binary: same annotated JPEG frames as the task's camera live feed (alias for `/cameras/{channelId}/live`). |

Requires `REDIS_URL` for live fan-out. Clients should reconnect after disconnect.

""".strip()



@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start one SSE bridge per process; stop on shutdown."""
    app.state.detection = detection
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
    description=_API_DESCRIPTION,
)
app.include_router(cashier_router, prefix="/cashier", tags=["Cashier Monitor"])
app.include_router(stream_metrics_router)


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


@app.post("/detection/stop/all")
def detection_stop_all():
    """Alias for stopping all cameras (same as ``POST /detection/stop`` with no ``camera_id``)."""
    return detection.on_post(DetectionRequest(action="stop_all", camera_id=None))


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
    channelId: Optional[str] = Query(
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

# ─────────────────────────────────────────────
# Live stream WebSocket routes
# ─────────────────────────────────────────────

@app.websocket("/cameras/{camera_id}/live")
async def camera_live_stream(websocket: WebSocket, camera_id: str):
    """
    Binary WebSocket stream of annotated JPEG frames for one camera.

    Each message is raw JPEG bytes — display in a browser with:

        const ws = new WebSocket("ws://host/cameras/cam1/live");
        ws.binaryType = "arraybuffer";
        ws.onmessage = e => {
            img.src = URL.createObjectURL(new Blob([e.data], {type:"image/jpeg"}));
        };
        ws.onclose = () => setTimeout(() => connect("cam1"), 2000);  // must reconnect manually

    Frames are dropped (never queued) when the client is slower than
    WS_SEND_TIMEOUT_MS (default 50 ms) — this prevents memory growth on
    slow or hidden browser tabs.

    Requires Redis (REDIS_URL). FrameBus publishes frames at REDIS_LIVE_FPS
    (default 13 fps) to keep bandwidth reasonable without visible quality loss.
    """
    await live_frames_ws(websocket, camera_id)


@app.websocket("/cameras/{camera_id}/events")
async def camera_events_stream(websocket: WebSocket, camera_id: str):
    """
    JSON WebSocket stream of detection events for one camera.

    Each message is a JSON string with the same shape as GET /detection/stream
    SSE events (eventType, taskId, timestamp, etc.).

    ws.onmessage = e => console.log(JSON.parse(e.data));
    ws.onclose   = () => setTimeout(() => connect("cam1"), 2000);
    """
    await live_events_ws(websocket, camera_id)


@app.websocket("/tasks/{task_name}/live")
async def task_live_stream(websocket: WebSocket, task_name: str):
    """
    Binary WebSocket stream of annotated JPEG frames, addressed by task name.

    This is a convenience alias for ``WS /cameras/{channelId}/live``. The server
    looks up the task by its exact ``taskName``, extracts the ``channelId``, and
    serves the same annotated frame stream that FrameBus publishes for that camera.

    Close codes returned before streaming begins:
      - 4004 — task name not found in the registry
      - 4009 — task name is ambiguous (shared by multiple tasks)
      - 1011 — Redis is unavailable

    Clients must implement a reconnect loop — the WebSocket does not
    auto-reconnect:

        const ws = new WebSocket("ws://host/tasks/mainentrance1/live");
        ws.binaryType = "arraybuffer";
        ws.onmessage = e => {
            img.src = URL.createObjectURL(new Blob([e.data], {type:"image/jpeg"}));
        };
        ws.onclose = () => setTimeout(() => connect(), 2000);
    """
    from fastapi import HTTPException as _HTTPException

    try:
        task = task_registry.require_by_name(task_name)
    except _HTTPException as exc:
        await websocket.accept()
        code = 4009 if exc.status_code == 409 else 4004
        await websocket.close(code=code, reason=exc.detail)
        return

    camera_id = str(task["channelId"])
    await live_frames_ws(websocket, camera_id)


# ─────────────────────────────────────────────
# ReID routes
# ─────────────────────────────────────────────
@app.post("/person_search/search")
async def person_search(file: UploadFile = File(...), top_k: int = Form(10)):
    return await person_search_api.search(file, top_k)


@app.get("/person_search/health")
def person_search_health():
    ready = getattr(person_search_api.person_search_service, "model", None) is not None
    return {"model_loaded": ready, "status": "ok" if ready else "unavailable"}


# ─────────────────────────────────────────────
# Semantic Search routes
# ─────────────────────────────────────────────
@app.post("/semantic_search/search")
async def semantic_search(
    text_query: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    top_k: int = Form(10),
):
    return await semantic_search_api.search(text_query=text_query, file=file, top_k=top_k)


@app.get("/semantic_search/health")
def semantic_search_health():
    ready = bool(getattr(semantic_search_api.semantic_search_service, "_ready", False))
    return {"model_loaded": ready, "status": "ok" if ready else "unavailable"}