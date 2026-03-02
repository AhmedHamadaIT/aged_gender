"""
app.py
------
FastAPI entry point.
Owns all routes and startup logic.

Run with:
  uvicorn app:app --host 0.0.0.0 --port 9000
"""

from fastapi import FastAPI
from schemas import DetectionRequest, DetectionStatus
from apis.detection import detection

app = FastAPI(title="YOLO Detection API", version="1.0.0")


# ─────────────────────────────────────────────
# Startup
# ─────────────────────────────────────────────
@app.on_event("startup")
def startup():
    print("[APP] Loading services...")
    detection.init()
    print("[APP] Ready.")


# ─────────────────────────────────────────────
# Detection routes
# ─────────────────────────────────────────────
@app.post("/detection")
def detection_post(req: DetectionRequest):
    return detection.on_post(req)


@app.get("/detection", response_model=DetectionStatus)
def detection_get():
    return detection.on_get()


# Add future services here:
# from apis.counting import counting
# @app.post("/counting")
# def counting_post(req: CountingRequest):
#     return counting.on_post(req)
# @app.get("/counting", response_model=CountingStatus)
# def counting_get():
#     return counting.on_get()