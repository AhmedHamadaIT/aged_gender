"""
schemas.py
----------
Pydantic models for request/response validation.
"""

from typing import Optional
from pydantic import BaseModel


class DetectionRequest(BaseModel):
    action: str  # "start" | "stop"


class DetectionStatus(BaseModel):
    running          : bool
    frame_count      : int
    fps              : float
    last_detections  : int
    total_detections : int
    uptime_seconds   : Optional[float]
    error            : Optional[str]