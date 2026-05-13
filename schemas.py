"""
schemas.py
----------
Pydantic models for request/response validation.
"""

from typing import Optional, Dict, List
from pydantic import BaseModel, ConfigDict


class DetectionRequest(BaseModel):
    action    : str
    camera_id : Optional[str] = None
    # When true, allow POST /detection/start without camera_id to start all channels
    all_channels: bool = False


class CameraStatus(BaseModel):
    model_config = ConfigDict(extra="ignore")

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
    # FrameBus live annotation / Redis (optional; present when detection is running)
    save_output               : Optional[bool] = None
    # Local/test: MP4 of the same annotated BGR as live Redis/WebSocket (see SAVE_ANNOTATED_VIDEO).
    save_annotated_video      : Optional[bool] = None
    annotated_video_path      : Optional[str] = None
    redis_connected           : Optional[bool] = None
    last_live_publish_seq     : Optional[int] = None
    last_live_frame_had_boxes : Optional[bool] = None
    live_annotation_mode      : Optional[str] = None
    live_jpeg_quality         : Optional[int] = None
    task_queue_jpeg_quality   : Optional[int] = None
    redis_circuit_state       : Optional[str] = None


class DetectionStatus(BaseModel):
    cameras: Dict[str, CameraStatus]
