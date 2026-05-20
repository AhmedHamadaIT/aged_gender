"""
apis/cameras.py
---------------
Camera registry resource.

Manages camera configuration in memory.
Cameras can be added/removed while detection is running.

Endpoints (registered in app.py):
    POST   /cameras                  → add one or more cameras
    GET    /cameras                  → list all configured cameras
    PATCH  /cameras/{cam_id}         → update RTSP URL for one camera
    DELETE /cameras/{cam_id}         → remove a camera
    POST   /cameras/{cam_id}/tasks   → ensure a task uses this camera (updates channelId if needed)
"""

import os
import re
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import cv2
from fastapi.responses import FileResponse
from fastapi import HTTPException
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

from utils.rtsp_ffmpeg import normalize_rtsp_source_url, open_rtsp_videocapture


# ─────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────
class CameraConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    url: str

    @field_validator("id", mode="before")
    @classmethod
    def _id_to_str(cls, v: Any) -> str:
        if v is None:
            raise TypeError("id is required")
        if isinstance(v, bool):
            raise TypeError("id must be str or int, not bool")
        return str(v)

    @field_validator("url", mode="before")
    @classmethod
    def _normalize_url(cls, v: Any) -> str:
        if v is None:
            raise TypeError("url is required")
        return normalize_rtsp_source_url(str(v))


class CameraSetupRequest(BaseModel):
    """Batch shape: ``{\"cameras\": [{\"id\", \"url\"}, ...]}``.

    Also accepts a **single flat** body (common from dashboards):
    ``{\"id\"|\"camera_id\", \"url\"|\"rtsp_url\", ...}`` — extra keys are ignored.
    """

    model_config = ConfigDict(extra="ignore")

    cameras: list[CameraConfig] = Field(..., min_length=1)

    @model_validator(mode="before")
    @classmethod
    def _coerce_flat_single_camera(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if data.get("cameras") is not None:
            return data
        cam_id = data.get("id", data.get("camera_id"))
        url = data.get("url", data.get("rtsp_url"))
        if cam_id is not None and url is not None:
            return {"cameras": [{"id": cam_id, "url": url}]}
        return data


class CameraPatchRequest(BaseModel):
    """Body for ``PATCH /cameras/{camera_id}`` — update stream URL only."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    url: str = Field(..., validation_alias=AliasChoices("url", "rtsp_url"))

    @field_validator("url", mode="before")
    @classmethod
    def _normalize_patch_url(cls, v: Any) -> str:
        if v is None:
            raise TypeError("url is required")
        return normalize_rtsp_source_url(str(v))


class CameraTaskLinkBody(BaseModel):
    """Body for ``POST /cameras/{camera_id}/tasks``; JSON may use ``taskId`` or ``task_id``."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    task_id: int = Field(..., validation_alias=AliasChoices("task_id", "taskId"))
    enable: Optional[bool] = None


def camera_link_task(cam_id: str, body: CameraTaskLinkBody) -> dict:
    """Point an existing task at this camera (``channelId``); optional ``enable`` toggle.

    Tasks are normally bound via ``channelId`` on ``POST /api/tasks``. This route is a
    convenience for clients that register the camera first, then attach tasks.
    """
    # Import here to avoid import cycles at module load.
    from apis.tasks import TaskConfig, task_registry

    cam_id_s = str(cam_id)
    if camera_registry.get(cam_id_s) is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No camera registered with id '{cam_id_s}'. "
                "Register it via POST /cameras first."
            ),
        )

    task_dict = task_registry.get(body.task_id)
    if task_dict is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No task with taskId {body.task_id}. "
                "Create it via POST /api/tasks first."
            ),
        )

    t = dict(task_dict)
    changed = False
    if str(t.get("channelId", "")) != cam_id_s:
        t["channelId"] = cam_id_s
        changed = True
    if body.enable is not None and bool(t.get("enable", True)) != body.enable:
        t["enable"] = body.enable
        changed = True

    if changed:
        task_registry.upsert(TaskConfig(**t))
        out_status = "updated"
    else:
        out_status = "ok"

    return {
        "status": out_status,
        "camera_id": cam_id_s,
        "task": task_registry.require(body.task_id),
    }


# ─────────────────────────────────────────────
# Camera registry
# ─────────────────────────────────────────────
class CameraRegistry:
    def __init__(self):
        self._cameras: Dict[str, str] = {}  # {cam_id: rtsp_url}
        self._snapshot_lock = threading.Lock()
        # Last good snapshot path + monotonic time when it was taken
        self._snapshot_cache: Dict[str, Tuple[Optional[str], float]] = {}
        self._snapshot_frame_index = max(1, int(os.getenv("CAMERA_SNAPSHOT_FRAME_INDEX", "5")))
        self._snapshot_jpeg_quality = min(
            100,
            max(1, int(os.getenv("CAMERA_SNAPSHOT_JPEG_QUALITY", "75"))),
        )
        self._snapshot_max_reads = max(self._snapshot_frame_index + 2, int(os.getenv("CAMERA_SNAPSHOT_MAX_READS", "15")))
        self._snapshot_cache_ttl = max(0.0, float(os.getenv("CAMERA_SNAPSHOT_CACHE_TTL_SEC", "5")))
        self._snapshot_dir = os.path.abspath(os.getenv("CAMERA_SNAPSHOT_DIR", "./outputs/camera_snapshots"))
        self._snapshot_base_url = os.getenv("CAMERA_SNAPSHOT_BASE_URL", "http://127.0.0.1:9000").rstrip("/")
        os.makedirs(self._snapshot_dir, exist_ok=True)

    def add(self, cam_id: str, url: str):
        self._cameras[cam_id] = normalize_rtsp_source_url(url)

    def remove(self, cam_id: str):
        if cam_id not in self._cameras:
            raise HTTPException(status_code=404, detail=f"Camera '{cam_id}' not found.")
        del self._cameras[cam_id]
        with self._snapshot_lock:
            self._snapshot_cache.pop(cam_id, None)

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
            snapshot_path = self._capture_snapshot(cam_id, url)
            snapshot_url = None
            if snapshot_path:
                snapshot_url = f"{self._snapshot_base_url}/snapshots/{os.path.basename(snapshot_path)}"
            cameras.append(
                {
                    "id": cam_id,
                    "url": url,
                    "snapshot": snapshot_url,
                }
            )
        return {
            "count"  : len(self._cameras),
            "cameras": cameras,
        }

    def on_snapshot_file_get(self, file_name: str):
        safe_name = os.path.basename(file_name)
        if safe_name != file_name:
            raise HTTPException(status_code=400, detail="Invalid snapshot filename.")
        path = os.path.join(self._snapshot_dir, safe_name)
        if not os.path.isfile(path):
            raise HTTPException(status_code=404, detail=f"Snapshot '{file_name}' not found.")
        return FileResponse(path=path, media_type="image/jpeg")

    def _capture_snapshot(self, cam_id: str, url: str) -> Optional[str]:
        """
        Return a JPEG path for this camera, using a short-lived cache to avoid
        opening a fresh RTSP session on every poll. On transient failure, returns
        the last successful snapshot path if any.
        """
        now = time.monotonic()
        with self._snapshot_lock:
            if cam_id in self._snapshot_cache and self._snapshot_cache_ttl > 0:
                path, ts = self._snapshot_cache[cam_id]
                if now - ts < self._snapshot_cache_ttl and path and os.path.isfile(path):
                    return path

        new_path: Optional[str] = None
        try:
            new_path = self._read_jpeg_from_source(cam_id, url)
        except Exception:  # noqa: BLE001
            new_path = None

        with self._snapshot_lock:
            if new_path:
                self._snapshot_cache[cam_id] = (new_path, time.monotonic())
                return new_path
            stale = self._snapshot_cache.get(cam_id, (None, 0.0))[0]
            if stale and os.path.isfile(stale):
                return stale
        return new_path

    def _read_jpeg_from_source(self, cam_id: str, url: str) -> Optional[str]:
        """
        Read one stable frame, save as JPEG, return path or None.
        """
        cap = None
        try:
            if url.startswith("rtsp://"):
                cap = open_rtsp_videocapture(url)
            else:
                cap = cv2.VideoCapture(url)

            if not cap.isOpened():
                return None

            valid_frame = None
            good_frames = 0
            for _ in range(self._snapshot_max_reads):
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                good_frames += 1
                if good_frames >= self._snapshot_frame_index:
                    valid_frame = frame
                    break

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
        finally:
            if cap is not None:
                cap.release()

    def on_patch(self, cam_id: str, req: CameraPatchRequest):
        cam_id_s = str(cam_id)
        if cam_id_s not in self._cameras:
            raise HTTPException(
                status_code=404,
                detail=f"Camera '{cam_id_s}' not found.",
            )
        self._cameras[cam_id_s] = req.url
        with self._snapshot_lock:
            self._snapshot_cache.pop(cam_id_s, None)
        return {
            "status": "updated",
            "camera_id": cam_id_s,
            "url": self._cameras[cam_id_s],
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
