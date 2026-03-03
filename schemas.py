"""
schemas.py
----------
Pydantic models for request/response validation.
"""

from typing import Optional, Dict
from pydantic import BaseModel


class DetectionRequest(BaseModel):
    action    : str            # "start" | "stop" | "stop_all"
    camera_id : Optional[str] = None  # None = all cameras


class CameraStatus(BaseModel):
    camera_id        : str
    rtsp_url         : str
    running          : bool
    frame_count      : int
    fps              : float
    last_detections  : int
    total_detections : int
    uptime_seconds   : Optional[float]
    error            : Optional[str]


class DetectionStatus(BaseModel):
    cameras: Dict[str, CameraStatus]