"""
services/face/face_task.py
---------------------------
Task configuration manager for face recognition.

Stores and manages FACE task configurations that define:
    - Which camera to monitor
    - Which face libraries to match against
    - Quality / pose thresholds for acceptance
    - Optional attribute extraction (age, gender, emotion)
    - Active schedule (weekdays + time range)

Matches the input spec:
    {
      "taskId": 9,
      "taskName": "attendance",
      "algorithmType": "FACE",
      "channelId": 9,
      "enable": true,
      "threshold": 70,
      "libIds": "-1",
      "enableStranger": true,
      "detailConfig": {
        "facePixelSize": 60,
        "model": "Fast",
        "yawThreshold": 35,
        "pitchThreshold": 25,
        "failCount": 2,
        "enableAgeGenderDetect": false,
        "enableEmotionDetect": false
      },
      "validWeekday": ["MONDAY","TUESDAY","WEDNESDAY","THURSDAY","FRIDAY"],
      "validStartTime": 0,
      "validEndTime": 86399000
    }
"""

import json
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

from logger.logger_config import Logger

log = Logger.get_logger(__name__)


# ─────────────────────────────────────────────
# Config data classes
# ─────────────────────────────────────────────
@dataclass
class DetailConfig:
    facePixelSize         : int  = 60
    model                 : str  = "Fast"      # "Fast" | "Accurate"
    yawThreshold          : int  = 35
    pitchThreshold        : int  = 25
    failCount             : int  = 2
    enableAgeGenderDetect : bool = False
    enableEmotionDetect   : bool = False

    @staticmethod
    def from_dict(d: dict) -> "DetailConfig":
        return DetailConfig(
            facePixelSize         = d.get("facePixelSize", 60),
            model                 = d.get("model", "Fast"),
            yawThreshold          = d.get("yawThreshold", 35),
            pitchThreshold        = d.get("pitchThreshold", 25),
            failCount             = d.get("failCount", 2),
            enableAgeGenderDetect = d.get("enableAgeGenderDetect", False),
            enableEmotionDetect   = d.get("enableEmotionDetect", False),
        )

    def to_dict(self) -> dict:
        return {
            "facePixelSize"         : self.facePixelSize,
            "model"                 : self.model,
            "yawThreshold"          : self.yawThreshold,
            "pitchThreshold"        : self.pitchThreshold,
            "failCount"             : self.failCount,
            "enableAgeGenderDetect" : self.enableAgeGenderDetect,
            "enableEmotionDetect"   : self.enableEmotionDetect,
        }


@dataclass
class FaceTaskConfig:
    taskId         : int
    taskName       : str
    algorithmType  : str  = "FACE"
    channelId      : int  = 0
    enable         : bool = True
    threshold      : int  = 70
    libIds         : str  = "-1"
    enableStranger : bool = True
    detailConfig   : DetailConfig = field(default_factory=DetailConfig)
    validWeekday   : List[str] = field(default_factory=lambda: [
        "MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY",
        "SATURDAY", "SUNDAY"
    ])
    validStartTime : int = 0           # ms from midnight
    validEndTime   : int = 86399000    # ms from midnight

    @staticmethod
    def from_dict(d: dict) -> "FaceTaskConfig":
        dc = d.get("detailConfig", {})
        return FaceTaskConfig(
            taskId         = d["taskId"],
            taskName       = d.get("taskName", ""),
            algorithmType  = d.get("algorithmType", "FACE"),
            channelId      = d.get("channelId", 0),
            enable         = d.get("enable", True),
            threshold      = d.get("threshold", 70),
            libIds         = d.get("libIds", "-1"),
            enableStranger = d.get("enableStranger", True),
            detailConfig   = DetailConfig.from_dict(dc) if isinstance(dc, dict) else DetailConfig(),
            validWeekday   = d.get("validWeekday", [
                "MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY",
                "SATURDAY", "SUNDAY"
            ]),
            validStartTime = d.get("validStartTime", 0),
            validEndTime   = d.get("validEndTime", 86399000),
        )

    def to_dict(self) -> dict:
        return {
            "taskId"         : self.taskId,
            "taskName"       : self.taskName,
            "algorithmType"  : self.algorithmType,
            "channelId"      : self.channelId,
            "enable"         : self.enable,
            "threshold"      : self.threshold,
            "libIds"         : self.libIds,
            "enableStranger" : self.enableStranger,
            "detailConfig"   : self.detailConfig.to_dict(),
            "validWeekday"   : self.validWeekday,
            "validStartTime" : self.validStartTime,
            "validEndTime"   : self.validEndTime,
        }


# ─────────────────────────────────────────────
# Task manager
# ─────────────────────────────────────────────
WEEKDAY_MAP = {
    "MONDAY": 0, "TUESDAY": 1, "WEDNESDAY": 2, "THURSDAY": 3,
    "FRIDAY": 4, "SATURDAY": 5, "SUNDAY": 6,
}


class FaceTaskManager:
    """
    Manages active face recognition tasks.

    Tasks can be created, updated, deleted, and queried.
    Supports schedule-based activation (weekday + time range).
    Persists tasks to a JSON file under the face storage dir.
    """

    def __init__(self, storage_dir: str = None):
        self._storage_dir = Path(storage_dir or os.getenv("FACE_STORAGE_DIR", "./data/face"))
        self._tasks_file  = self._storage_dir / "tasks.json"
        self._tasks: Dict[int, FaceTaskConfig] = {}
        self._lock = threading.Lock()

        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self):
        """Load tasks from disk."""
        if self._tasks_file.exists():
            try:
                data = json.loads(self._tasks_file.read_text())
                for td in data:
                    task = FaceTaskConfig.from_dict(td)
                    self._tasks[task.taskId] = task
                log.info(f"[FACE_TASK] Loaded {len(self._tasks)} tasks")
            except Exception as e:
                log.warning(f"[FACE_TASK] Failed to load tasks: {e}")

    def _save(self):
        """Persist all tasks to disk."""
        data = [t.to_dict() for t in self._tasks.values()]
        self._tasks_file.write_text(json.dumps(data, indent=2))

    def create_task(self, config: dict) -> FaceTaskConfig:
        """Create or update a task configuration."""
        with self._lock:
            task = FaceTaskConfig.from_dict(config)
            self._tasks[task.taskId] = task
            self._save()
            log.info(f"[FACE_TASK] Created/updated task {task.taskId}: {task.taskName}")
            return task

    def get_task(self, task_id: int) -> Optional[FaceTaskConfig]:
        with self._lock:
            return self._tasks.get(task_id)

    def delete_task(self, task_id: int) -> bool:
        with self._lock:
            if task_id not in self._tasks:
                return False
            del self._tasks[task_id]
            self._save()
            log.info(f"[FACE_TASK] Deleted task {task_id}")
            return True

    def list_tasks(self) -> List[dict]:
        with self._lock:
            return [t.to_dict() for t in self._tasks.values()]

    def active_tasks(self) -> List[FaceTaskConfig]:
        """
        Return tasks that are currently enabled and within their schedule window.
        """
        with self._lock:
            now    = datetime.now(timezone.utc)
            result = []

            for task in self._tasks.values():
                if not task.enable:
                    continue
                if not self._is_in_schedule(task, now):
                    continue
                result.append(task)

            return result

    @staticmethod
    def _is_in_schedule(task: FaceTaskConfig, now: datetime) -> bool:
        """Check if 'now' falls within the task's valid schedule."""
        # Weekday check
        weekday_name = now.strftime("%A").upper()
        if weekday_name not in task.validWeekday:
            return False

        # Time check (ms from midnight)
        ms_from_midnight = (
            now.hour * 3600 * 1000
            + now.minute * 60 * 1000
            + now.second * 1000
            + now.microsecond // 1000
        )
        if ms_from_midnight < task.validStartTime or ms_from_midnight > task.validEndTime:
            return False

        return True
