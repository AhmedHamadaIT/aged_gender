"""
schemas.py
----------
Pydantic models for request/response validation.
"""

from typing import Optional, Dict, List
from pydantic import BaseModel


class DetectionRequest(BaseModel):
    action    : str
    camera_id : Optional[str] = None


class CameraStatus(BaseModel):
    camera_id        : str
    rtsp_url         : str
    running          : bool
    frame_count      : int
    fps              : float
    last_detections  : int
    total_detections : int
    uptime_seconds   : Optional[float] = None
    error            : Optional[str]   = None


class DetectionStatus(BaseModel):
    cameras: Dict[str, CameraStatus]
