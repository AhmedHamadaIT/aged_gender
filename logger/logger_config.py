"""
logger/logger_config.py
-----------------------
M-15: Shared logger with optional JSON structured output and contextvars support.

Environment variables:
    LOG_LEVEL   — DEBUG | INFO | WARNING | ERROR (default INFO)
    LOG_FORMAT  — "json" or "text" (default "text")
    LOG_APP_FILE — path to log file (default ./logger/app.log)

When LOG_FORMAT=json every log line is a single JSON object:
    {
        "ts"       : "2026-05-23T00:00:00.000Z",
        "level"    : "INFO",
        "logger"   : "frame_bus",
        "msg"      : "...",
        "camera_id": "cam1",   ← populated from contextvar when set
        "task_id"  : "7",      ← populated from contextvar when set
        "frame_id" : 1234      ← populated from contextvar when set
    }
"""

import json
import logging
import os
from contextvars import ContextVar
from datetime import datetime, timezone

# ── Public context variables ───────────────────────────────────────────────────
# Set these at the start of a FrameBus/TaskWorker processing loop to enrich all
# log lines emitted during that iteration without passing values through callers.

ctx_camera_id: ContextVar[str] = ContextVar("camera_id", default="")
ctx_task_id:   ContextVar[str] = ContextVar("task_id",   default="")
ctx_frame_id:  ContextVar[int] = ContextVar("frame_id",  default=0)


def _resolve_log_level() -> int:
    """Read LOG_LEVEL env (e.g. DEBUG, INFO, WARNING) and return the logging int level."""
    raw = os.getenv("LOG_LEVEL", "INFO").upper().strip()
    return getattr(logging, raw, logging.INFO)


class _JsonFormatter(logging.Formatter):
    """Emit one compact JSON object per log record, including context vars."""

    def format(self, record: logging.LogRecord) -> str:
        record.message = record.getMessage()
        payload: dict = {
            "ts"    : datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level" : record.levelname,
            "logger": record.name,
            "msg"   : record.message,
        }
        cam = ctx_camera_id.get("")
        if cam:
            payload["camera_id"] = cam
        tid = ctx_task_id.get("")
        if tid:
            payload["task_id"] = tid
        fid = ctx_frame_id.get(0)
        if fid:
            payload["frame_id"] = fid
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


class Logger:
    _logger = None

    @classmethod
    def get_logger(cls, name):
        if cls._logger is None:
            level = _resolve_log_level()
            cls._logger = logging.getLogger("shared_logger")
            cls._logger.setLevel(level)

            if not cls._logger.hasHandlers():
                log_file_path = (
                    os.getenv("LOG_APP_FILE", "./logger/app.log").strip()
                    or "./logger/app.log"
                )
                os.makedirs(os.path.dirname(log_file_path) or ".", exist_ok=True)

                use_json = os.getenv("LOG_FORMAT", "text").strip().lower() == "json"

                if use_json:
                    formatter = _JsonFormatter()
                else:
                    formatter = logging.Formatter(
                        "%(asctime)s - %(name)s - %(levelname)s - "
                        "[%(filename)s:%(lineno)d] - %(message)s"
                    )

                file_handler    = logging.FileHandler(log_file_path)
                console_handler = logging.StreamHandler()

                file_handler.setLevel(level)
                console_handler.setLevel(level)

                file_handler.setFormatter(formatter)
                console_handler.setFormatter(formatter)

                cls._logger.addHandler(file_handler)
                cls._logger.addHandler(console_handler)

        return cls._logger.getChild(name)
