"""
apis/cameras.py
---------------
Camera registry resource.

Manages camera configuration in memory.
Cameras can be added/removed while detection is running.

Endpoints (registered in app.py):
    POST   /cameras          → add one or more cameras
    GET    /cameras          → list all configured cameras
    DELETE /cameras/{cam_id} → remove a camera
"""

import os
import re
from datetime import datetime, timezone
from typing import Dict, Optional

import cv2
from fastapi import HTTPException
from pydantic import BaseModel, Field

import stream


# ─────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────
class CameraConfig(BaseModel):
    id : str
    url: str


class CameraSetupRequest(BaseModel):
    cameras: list[CameraConfig] = Field(..., min_length=1)


# ─────────────────────────────────────────────
# Camera registry
# ─────────────────────────────────────────────
class CameraRegistry:
    def __init__(self):
        self._cameras: Dict[str, str] = {}  # {cam_id: rtsp_url}
        self._snapshot_frame_index = max(1, int(os.getenv("CAMERA_SNAPSHOT_FRAME_INDEX", "5")))
        self._snapshot_jpeg_quality = min(
            100,
            max(1, int(os.getenv("CAMERA_SNAPSHOT_JPEG_QUALITY", "75"))),
        )
        self._snapshot_max_reads = max(self._snapshot_frame_index + 2, int(os.getenv("CAMERA_SNAPSHOT_MAX_READS", "15")))
        self._snapshot_dir = os.path.abspath(os.getenv("CAMERA_SNAPSHOT_DIR", "./outputs/camera_snapshots"))
        os.makedirs(self._snapshot_dir, exist_ok=True)

    def add(self, cam_id: str, url: str):
        self._cameras[cam_id] = url

    def remove(self, cam_id: str):
        if cam_id not in self._cameras:
            raise HTTPException(status_code=404, detail=f"Camera '{cam_id}' not found.")
        del self._cameras[cam_id]

    def get(self, cam_id: str) -> Optional[str]:
        return self._cameras.get(cam_id)

    def all(self) -> Dict[str, str]:
        return dict(self._cameras)

    def ids(self) -> list:
        return list(self._cameras.keys())

    def on_post(self, req: CameraSetupRequest):
        for cam in req.cameras:
            self.add(cam.id, cam.url)
        return {
            "status" : "configured",
            "cameras": self.all(),
        }

    def on_get(self):
        cameras = []
        for cam_id, url in self._cameras.items():
            cameras.append(
                {
                    "id": cam_id,
                    "url": url,
                    "snapshot": self._capture_snapshot(cam_id, url),
                }
            )
        return {
            "count"  : len(self._cameras),
            "cameras": cameras,
        }

    def _capture_snapshot(self, cam_id: str, url: str) -> Optional[str]:
        """
        Read one stable frame via ``stream.capture_preview_frame`` (same path as
        detection / H265-capable RTSP) and save JPEG. Returns None on failure.
        """
        try:
            valid_frame = stream.capture_preview_frame(
                url,
                max_reads=self._snapshot_max_reads,
                settle_after_reads=self._snapshot_frame_index,
            )
            if valid_frame is None:
                return None

            safe_cam_id = re.sub(r"[^a-zA-Z0-9_.-]+", "_", cam_id).strip("_") or "camera"
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
            file_name = f"{safe_cam_id}_{ts}.jpg"
            file_path = os.path.join(self._snapshot_dir, file_name)

            ok = cv2.imwrite(
                file_path,
                valid_frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), self._snapshot_jpeg_quality],
            )
            if not ok:
                return None

            return file_path
        except Exception:
            return None

    def on_delete(self, cam_id: str):
        self.remove(cam_id)
        return {
            "status"  : "removed",
            "camera_id": cam_id,
            "remaining": self.ids(),
        }


# ── Singleton ─────────────────────────────────
camera_registry = CameraRegistry()