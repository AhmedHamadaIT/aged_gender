"""
apis/tasks.py
-------------
Task registry — stores and manages task configurations from the backend.

A task defines:
  - which algorithm to run (algorithmType)
  - which camera to run it on (channelId)
  - algorithm-specific config (lines, threshold, schedule, etc.)

Endpoints (registered in app.py):
    POST   /api/tasks             → create / upsert a task
    GET    /api/tasks             → list all tasks
    GET    /api/tasks/{task_id}   → get one task
    PUT    /api/tasks/{task_id}   → update a task
    DELETE /api/tasks/{task_id}   → remove a task
"""

from __future__ import annotations

import json
import logging
import os
import signal
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from pydantic import BaseModel, field_validator

_log_tasks = logging.getLogger(__name__)

# Sentinel returned when a taskName lookup is ambiguous (multiple tasks share the same name).
_AMBIGUOUS = object()


# ─────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────
class DetailConfig(BaseModel):
    # CROSS_LINE
    enableAttrDetect: bool       = False
    enableReid      : bool       = False
    # MASK_HAIRNET_CHEF_HAT
    alarmType       : List[str]  = []
    # CASHIER_BOX_OPEN
    drawerOpenLimit : int        = 30
    serviceWaitLimit: int        = 30
    enableStaffList : bool       = False
    staffIds        : List[int]  = []
    # FACE
    facePixelSize   : int        = 60
    qualityThreshold: int        = 60
    yawThreshold    : int        = 35
    pitchThreshold  : int        = 25
    failCount       : int        = 2
    # QW-7: minimum bbox area (px²) — 0 = disabled (all tasks)
    minPersonAreaPx : int        = 0
    # QW-8: cross-line cooldown seconds per (track_id, line_id) — 0 = disabled
    debounceSec     : float      = 0.0
    # M-4: per-task confidence threshold override — 0.0 = use global CONF_THRESHOLD
    confThreshold   : float      = 0.0


class TaskConfig(BaseModel):
    taskId        : int
    taskName      : str
    algorithmType : str
    channelId     : str
    enable        : bool        = True
    threshold     : int         = 50
    areaPosition  : str         = "[]"
    detailConfig  : DetailConfig = DetailConfig()
    validWeekday  : List[str]   = [
        "MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"
    ]
    validStartTime: int  = 0
    validEndTime  : int  = 86400000   # end of day in ms
    # FACE-specific (ignored by other tasks)
    libIds        : str         = "-1"
    enableStranger: bool        = True

    @field_validator("channelId", mode="before")
    @classmethod
    def _channel_id_to_str(cls, v: Any) -> str:
        """Allow numeric JSON (e.g. 1) while matching cameras by string id everywhere."""
        if v is None:
            raise TypeError("channelId is required")
        if isinstance(v, bool):
            raise TypeError("channelId must be str or int, not bool")
        return str(v)


# ─────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────
def _validate_cross_line_area_position(area_position: str) -> None:
    """
    Enabled CROSS_LINE tasks must have areaPosition that yields at least one
    effective line — same rules as ``CrossLineTask`` /
    ``parse_effective_cross_lines`` (see services/cross_line.py).
    """
    from services.cross_line import parse_effective_cross_lines

    if not area_position or not str(area_position).strip():
        raise HTTPException(
            status_code=400,
            detail=(
                "CROSS_LINE task with enable=true requires a non-empty areaPosition "
                "JSON array with at least one valid line (two {x,y} points per line)."
            ),
        )
    try:
        parsed: Any = json.loads(area_position)
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=400,
            detail=f"areaPosition must be valid JSON: {e}",
        ) from e
    if not isinstance(parsed, list) or len(parsed) < 1:
        raise HTTPException(
            status_code=400,
            detail=(
                "CROSS_LINE areaPosition must be a JSON array with at least one line object."
            ),
        )
    if len(parse_effective_cross_lines(area_position)) < 1:
        raise HTTPException(
            status_code=400,
            detail=(
                "CROSS_LINE areaPosition must contain at least one valid line: "
                "each line needs \"point\" with two {x,y} objects with numeric coordinates."
            ),
        )


# ─────────────────────────────────────────────
# M-5: Cross-line line patch
# ─────────────────────────────────────────────

class CrossLineLinePatch(BaseModel):
    """Partial update for a single virtual line within a CROSS_LINE task."""
    point    : Optional[List[Dict[str, Any]]] = None
    direction: Optional[int]                  = None


def _refresh_cross_line_stream(task_id: int, task: dict) -> dict:
    """
    Ask the running DetectionResource to hot-reload the cross-line task.
    Returns {"applied": bool, "reason": str, ...}.
    Late import avoids circular dependency.
    """
    try:
        from apis.detection import detection  # noqa: PLC0415
        if detection is not None:
            return detection.reload_cross_line_task(task_id, task)
    except Exception:
        pass
    return {"applied": False, "reason": "detection_not_running"}


def _checkpoint_path() -> Path:
    """Absolute path to the registry checkpoint file."""
    out_dir = os.getenv("OUTPUT_DIR", "/output")
    return Path(out_dir) / "registry_checkpoint.json"


def _save_checkpoint(tasks: dict) -> None:
    """Atomically write all task configs to JSON checkpoint."""
    cp = _checkpoint_path()
    try:
        cp.parent.mkdir(parents=True, exist_ok=True)
        tmp = cp.with_suffix(".tmp")
        tmp.write_text(json.dumps(list(tasks.values()), indent=2), encoding="utf-8")
        tmp.replace(cp)
    except Exception as exc:
        _log_tasks.warning("registry_checkpoint: save failed: %s", exc)


def _load_checkpoint() -> Optional[dict]:
    """Load checkpoint and return {task_id: config} or None."""
    cp = _checkpoint_path()
    if not cp.exists():
        return None
    try:
        data = json.loads(cp.read_text(encoding="utf-8"))
        return {int(t["taskId"]): t for t in data}
    except Exception as exc:
        _log_tasks.warning("registry_checkpoint: load failed: %s", exc)
        return None


class TaskRegistry:
    SUPPORTED = {"CROSS_LINE", "MASK_HAIRNET_CHEF_HAT", "CASHIER_BOX_OPEN", "PHONE_USAGE", "FACE"}

    def __init__(self):
        self._tasks: dict = {}   # {task_id (int): task_config (dict)}

        # M-7: restore from checkpoint on boot when REGISTRY_RESTORE=true.
        if os.getenv("REGISTRY_RESTORE", "false").lower() in ("true", "1", "yes"):
            restored = _load_checkpoint()
            if restored:
                self._tasks = restored
                _log_tasks.info(
                    "registry_checkpoint: restored %d tasks from %s",
                    len(restored), _checkpoint_path(),
                )

        # M-7: save on SIGTERM so checkpoint is current on graceful shutdown.
        try:
            signal.signal(signal.SIGTERM, self._sigterm_handler)
        except (ValueError, OSError):
            pass  # Non-main thread — SIGTERM handler set elsewhere.

    def _sigterm_handler(self, signum, frame) -> None:
        _save_checkpoint(self._tasks)

    def _validate_config(self, config: TaskConfig) -> None:
        if config.algorithmType not in self.SUPPORTED:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported algorithmType '{config.algorithmType}'. "
                    f"Supported: {sorted(self.SUPPORTED)}"
                ),
            )
        if config.algorithmType == "CROSS_LINE" and config.enable:
            _validate_cross_line_area_position(config.areaPosition)

    # ── CRUD ──────────────────────────────────

    def upsert(self, config: TaskConfig) -> dict:
        self._validate_config(config)
        self._tasks[config.taskId] = config.model_dump()
        _save_checkpoint(self._tasks)
        return self._tasks[config.taskId]

    def get(self, task_id: int) -> Optional[dict]:
        return self._tasks.get(task_id)

    def require(self, task_id: int) -> dict:
        task = self.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found.")
        return task

    def remove(self, task_id: int):
        self.require(task_id)
        del self._tasks[task_id]
        _save_checkpoint(self._tasks)

    def all(self) -> list:
        return list(self._tasks.values())

    def get_enabled(self) -> list:
        return [t for t in self._tasks.values() if t.get("enable", True)]

    def get_by_name(self, task_name: str) -> Optional[dict]:
        """Return the task config for *task_name* (exact match).

        Returns ``None`` when no task matches.
        Raises :class:`fastapi.HTTPException` 409 when more than one task
        shares the same ``taskName`` — the caller cannot know which camera to
        use in that case.
        """
        matches = [t for t in self._tasks.values() if t.get("taskName") == task_name]
        if not matches:
            return None
        if len(matches) > 1:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"taskName '{task_name}' is shared by {len(matches)} tasks "
                    f"(ids: {[m['taskId'] for m in matches]}). "
                    "Use a unique taskName or connect via /cameras/{camera_id}/live instead."
                ),
            )
        return matches[0]

    def require_by_name(self, task_name: str) -> dict:
        """Like ``get_by_name`` but raises 404 when the task does not exist."""
        task = self.get_by_name(task_name)
        if task is None:
            raise HTTPException(
                status_code=404,
                detail=f"No task with taskName '{task_name}' found.",
            )
        return task

    # ── API handlers ───────────────────────────

    def on_post(self, config: TaskConfig):
        existed = config.taskId in self._tasks
        task = self.upsert(config)
        _notify_detection_task_changed(
            config.taskId, enabled=bool(task.get("enable", True)), exists=True
        )
        return {"status": "updated" if existed else "created", "task": task}

    def on_get_all(self):
        return {"count": len(self._tasks), "tasks": self.all()}

    def on_get_one(self, task_id: int):
        return self.require(task_id)

    def on_put(self, task_id: int, config: TaskConfig):
        if config.taskId != task_id:
            raise HTTPException(
                status_code=400,
                detail="taskId in body must match the URL parameter."
            )
        self.require(task_id)
        task = self.upsert(config)
        _notify_detection_task_changed(
            task_id, enabled=bool(task.get("enable", True)), exists=True
        )
        return {"status": "updated", "task": task}

    def on_delete(self, task_id: int):
        self.remove(task_id)
        _notify_detection_task_changed(task_id, enabled=False, exists=False)
        return {"status": "deleted", "taskId": task_id}

    def on_patch_cross_line(
        self, task_id: int, line_id: str, patch: CrossLineLinePatch
    ) -> dict:
        """
        Update a single virtual line geometry within a CROSS_LINE task.
        Persists the change to the in-memory registry and asks DetectionResource
        to hot-reload the worker process (overlay + re-spawn) without restarting
        the whole camera pipeline.
        """
        task = self.require(task_id)
        if task.get("algorithmType") != "CROSS_LINE":
            raise HTTPException(
                status_code=400,
                detail=f"Task {task_id} is not a CROSS_LINE task.",
            )

        from services.cross_line import update_line_in_area_position

        try:
            new_ap = update_line_in_area_position(
                task.get("areaPosition", "[]"),
                line_id,
                point=patch.point,
                direction=patch.direction,
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        task["areaPosition"] = new_ap
        self._tasks[task_id] = task
        _save_checkpoint(self._tasks)

        refresh_result = _refresh_cross_line_stream(task_id, task)
        return {
            "status"        : "updated",
            "task_id"       : task_id,
            "line_id"       : line_id,
            "areaPosition"  : new_ap,
            "live_reload"   : refresh_result,
        }


def _notify_detection_task_changed(
    task_id: int, *, enabled: bool, exists: bool
) -> None:
    """
    Inform the running DetectionResource (if any) that a task was modified or
    deleted.  This updates the task_validity_map polled by task worker processes
    so they can stop cleanly without waiting for a full watchdog respawn.

    Uses a late import to avoid circular imports between apis.tasks and
    apis.detection at module load time.
    """
    try:
        from apis.detection import detection  # noqa: PLC0415
        if detection is not None:
            detection.update_task_validity(str(task_id), enabled=enabled, exists=exists)
    except Exception:
        pass  # Non-critical: worker will detect on next validity poll cycle


# ── Singleton ─────────────────────────────────
task_registry = TaskRegistry()
