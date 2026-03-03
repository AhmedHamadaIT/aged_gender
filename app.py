"""
app.py
------
Application entry point.
Owns all routes, startup, and pipeline composition.

To add a new service: import it and add pipeline.register() below.

Run with:
  uvicorn app:app --host 0.0.0.0 --port 9000
"""

from fastapi import FastAPI
from pipeline import CameraPipeline
from schemas import DetectionRequest, DetectionStatus
from apis.detection import detection
from services.detector import DetectorService

# Register future services here:
# from services.counter import CounterService
# from services.tracker import TrackerService

app = FastAPI(title="YOLO Detection API", version="1.0.0")


# ─────────────────────────────────────────────
# Pipeline factory
# Called once per camera process on start
# ─────────────────────────────────────────────
def build_pipeline(camera_id, rtsp_url, shared_state, stop_event):
    pipeline = CameraPipeline(camera_id, rtsp_url, shared_state, stop_event)

    # ── Register services in order ────────────
    pipeline.register(DetectorService)
    # pipeline.register(CounterService)
    # pipeline.register(TrackerService)

    pipeline.run()


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
    return {"service": "YOLO Detection API", "version": "1.0.0"}


# ─────────────────────────────────────────────
# Detection routes
# ─────────────────────────────────────────────
@app.post("/detection")
def detection_post(req: DetectionRequest):
    return detection.on_post(req)


@app.get("/detection", response_model=DetectionStatus)
def detection_get():
    return detection.on_get()