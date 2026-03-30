"""
app.py
------
Application entry point.
Owns all routes, startup, and pipeline composition.

Typical usage from a web app:
    1. POST /cameras               → configure cameras
    2. POST /detection/setup       → configure pipeline services
    3. POST /detection/start       → start pipeline
    4. GET  /detection/stream/{id} → SSE stream per camera (parallel)
    5. GET  /detection/status      → monitor camera status
    6. POST /detection/stop        → stop pipeline

Run with:
    uvicorn app:app --host 0.0.0.0 --port 9000
"""

import asyncio
import json
import os

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from apis.cameras import camera_registry, CameraSetupRequest
from apis.test_responses import get_test_responses
from apis.detection import detection, DetectionSetupRequest
from error_codes.error_codes import ErrorCode
from error_codes.response import error
from pipeline import CameraPipeline
from schemas import DetectionStatus

app = FastAPI(title="Vision Pipeline API", version="1.0.0")


# ─────────────────────────────────────────────
# Pipeline factory — called once per camera process on start
# ─────────────────────────────────────────────
def build_pipeline(camera_id, rtsp_url, shared_state, stop_event, result_queue, pipeline_names):
    from services import REGISTRY
    cam_pipeline = CameraPipeline(
        camera_id, rtsp_url, shared_state, stop_event, result_queue, pipeline_names
    )
    for name in pipeline_names:
        cam_pipeline.register(REGISTRY[name])
    cam_pipeline.run()


# ─────────────────────────────────────────────
# Startup
# ─────────────────────────────────────────────
@app.on_event("startup")
def startup():
    print("[APP] Configuring pipeline...")
    detection.set_pipeline(build_pipeline)
    print("[APP] Ready.")


# ─────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────
@app.get("/")
def root():
    return {"service": "Vision Pipeline API", "version": "1.0.0"}


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
# Detection routes
# ─────────────────────────────────────────────
@app.post("/detection/setup")
def detection_setup(req: DetectionSetupRequest):
    return detection.on_setup(req)


@app.post("/detection/start")
def detection_start(camera_id: str = None):
    return detection.on_post("start", camera_id)


@app.post("/detection/stop")
def detection_stop(camera_id: str = None):
    return detection.on_post("stop", camera_id)


@app.get("/detection/status", response_model=DetectionStatus)
def detection_status():
    return detection.on_get()


# ─────────────────────────────────────────────
# Test route — remove in production
# ─────────────────────────────────────────────
@app.get("/test/responses")
def test_responses():
    return get_test_responses()


@app.get("/detection/stream/{cam_id}")
async def detection_stream(cam_id: str):
    """
    SSE stream per camera. Open one connection per camera for parallel streaming.

    Each event:
    {
        "camera_id"  : "cam1",
        "frame_count": 42,
        "timestamp"  : "2026-03-08T11:29:47.123456",
        "frame"      : "<base64 JPEG>",
        "data": {
            "detection": {"count": 2, "items": [...]},
            "use_case" : {"age_gender": [...]}
        }
    }
    """
    result_queue, err = detection.get_result_queue(cam_id)
    if err:
        return error(err, detail=cam_id)

    async def event_generator():
        while True:
            try:
                result = result_queue.get_nowait()
                yield f"data: {json.dumps(result)}\n\n"
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