"""
app.py
------
Application entry point — owns all routes and startup.

Workflow:
    1. POST /cameras                  → register cameras (id → rtsp_url)
    2. POST /api/tasks                → register tasks (algorithmType, channelId, config)
    3. POST /detection/start          → start processing
    4. GET  /detection/stream         → SSE stream of crossing events
    5. GET  /detection/status         → monitor camera status
    6. POST /detection/stop           → stop processing

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

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from apis.cameras   import camera_registry, CameraSetupRequest
from apis.detection import detection
from apis.tasks     import task_registry, TaskConfig
from schemas        import DetectionRequest, DetectionStatus

app = FastAPI(title="Vision Pipeline API", version="2.0.0")


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
async def detection_stream():
    """
    SSE stream — emits one JSON event per line crossing detected across all cameras.
    Events are also persisted locally (JSONL + images) by each task worker.
    """
    result_queue = detection.result_queue()

    async def event_generator():
        while True:
            try:
                event = result_queue.get_nowait()
                yield f"data: {json.dumps(event)}\n\n"
            except Exception:
                await asyncio.sleep(0.01)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control"              : "no-cache",
            "X-Accel-Buffering"          : "no",
            "Access-Control-Allow-Origin": "*",
        },
    )
