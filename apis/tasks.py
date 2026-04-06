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

from typing import List, Optional
from fastapi import HTTPException
from pydantic import BaseModel


# ─────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────
class DetailConfig(BaseModel):
    # CROSS_LINE
    enableAttrDetect: bool       = False
    enableReid      : bool       = False
    # MASK_HAIRNET_CHEF_HAT
    alarmType       : List[str]  = []


class TaskConfig(BaseModel):
    taskId        : int
    taskName      : str
    algorithmType : str
    channelId     : int
    enable        : bool        = True
    threshold     : int         = 50
    areaPosition  : str         = "[]"
    detailConfig  : DetailConfig = DetailConfig()
    validWeekday  : List[str]   = [
        "MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"
    ]
    validStartTime: int  = 0
    validEndTime  : int  = 86400000   # end of day in ms


# ─────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────
class TaskRegistry:
    SUPPORTED = {"CROSS_LINE", "MASK_HAIRNET_CHEF_HAT", "CASHIER_DRAWER"}

    def __init__(self):
        self._tasks: dict = {}   # {task_id (int): task_config (dict)}

    # ── CRUD ──────────────────────────────────

    def upsert(self, config: TaskConfig) -> dict:
        if config.algorithmType not in self.SUPPORTED:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported algorithmType '{config.algorithmType}'. "
                    f"Supported: {sorted(self.SUPPORTED)}"
                ),
            )
        self._tasks[config.taskId] = config.model_dump()
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

    def all(self) -> list:
        return list(self._tasks.values())

    def get_enabled(self) -> list:
        return [t for t in self._tasks.values() if t.get("enable", True)]

    # ── API handlers ───────────────────────────

    def on_post(self, config: TaskConfig):
        task = self.upsert(config)
        return {"status": "created", "task": task}

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
        task = self.upsert(config)
        return {"status": "updated", "task": task}

    def on_delete(self, task_id: int):
        self.remove(task_id)
        return {"status": "deleted", "taskId": task_id}


# ── Singleton ─────────────────────────────────
task_registry = TaskRegistry()
