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
    # When true, allow POST /detection/start without camera_id to start all channels
    all_channels: bool = False


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
    # Reconciled in API layer: FrameBus process handle vs last shared_state update
    framebus_process_alive: Optional[bool] = None
    stopped_reason         : Optional[str] = None
    last_state_update_age_sec: Optional[float] = None


class DetectionStatus(BaseModel):
    cameras: Dict[str, CameraStatus]
