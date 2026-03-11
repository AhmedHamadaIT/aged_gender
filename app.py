"""
app.py
------
Application entry point.
Owns all routes, startup, and pipeline composition.

Pipeline services are driven entirely by .env:
    PIPELINE=detector,age_gender

To add a new service:
    1. Create services/your_service.py
    2. Add to REGISTRY in services/__init__.py
    3. Add name to PIPELINE in .env
    4. Restart container

Run with:
    uvicorn app:app --host 0.0.0.0 --port 9000
"""

import os
from fastapi import FastAPI
from pipeline import CameraPipeline
from schemas import DetectionRequest, DetectionStatus
from apis.detection import detection


from logger.logger_config import Logger
import os
from dotenv import load_dotenv
load_dotenv()
log = Logger.get_logger(__name__)


app = FastAPI(title="Vision Pipeline API", version="1.0.0")


# ─────────────────────────────────────────────
# Pipeline factory — called once per camera process on start
# ─────────────────────────────────────────────
def build_pipeline(camera_id, rtsp_url, shared_state, stop_event):
    from services import REGISTRY

    names        = [n.strip() for n in os.getenv("PIPELINE").split(",")]
    cam_pipeline = CameraPipeline(camera_id, rtsp_url, shared_state, stop_event)

    for name in names:
        if name not in REGISTRY:
            raise ValueError(
                f"[APP] Unknown service '{name}'. "
                f"Available: {list(REGISTRY.keys())}"
            )
        cam_pipeline.register(REGISTRY[name])

    cam_pipeline.run()


# ─────────────────────────────────────────────
# Startup
# ─────────────────────────────────────────────
@app.on_event("startup")
def startup():
    print("[APP] Configuring pipeline...")
    detection.set_pipeline(build_pipeline)
    print(f"[APP] Services : {os.getenv('PIPELINE', 'detector')}")
    print("[APP] Ready.")


# ─────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service" : "Vision Pipeline API",
        "version" : "1.0.0",
        "pipeline": os.getenv("PIPELINE", "detector").split(","),
    }


# ─────────────────────────────────────────────
# Detection routes
# ─────────────────────────────────────────────
@app.post("/detection")
def detection_post(req: DetectionRequest):
    return detection.on_post(req)


@app.get("/detection", response_model=DetectionStatus)
def detection_get():
    return detection.on_get()