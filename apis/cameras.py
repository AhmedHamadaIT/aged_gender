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

from typing import Dict, Optional
from fastapi import HTTPException
from pydantic import BaseModel


# ─────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────
class CameraConfig(BaseModel):
    id : str
    url: str


class CameraSetupRequest(BaseModel):
    cameras: list[CameraConfig]


# ─────────────────────────────────────────────
# Camera registry
# ─────────────────────────────────────────────
class CameraRegistry:
    def __init__(self):
        self._cameras: Dict[str, str] = {}  # {cam_id: rtsp_url}

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
        return {
            "count"  : len(self._cameras),
            "cameras": [{"id": k, "url": v} for k, v in self._cameras.items()],
        }

    def on_delete(self, cam_id: str):
        self.remove(cam_id)
        return {
            "status"  : "removed",
            "camera_id": cam_id,
            "remaining": self.ids(),
        }


# ── Singleton ─────────────────────────────────
camera_registry = CameraRegistry()