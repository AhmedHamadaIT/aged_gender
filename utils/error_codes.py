"""
utils/error_codes.py
--------------------
Canonical error codes and structured error types for the stream annotation
pipeline.  All validation failures, runtime stops, and worker errors MUST
reference a code from this module so downstream consumers (logs, SSE clients,
monitoring) can filter and alert by category.

Usage:
    from utils.error_codes import STREAM_UNREACHABLE, ValidationFailure
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

# ── Validation stage codes (pre-spawn) ───────────────────────────────────────

STREAM_NOT_REGISTERED  = "STREAM_NOT_REGISTERED"
"""Camera ID is not registered in the camera registry."""

STREAM_UNREACHABLE     = "STREAM_UNREACHABLE"
"""RTSP/HTTP stream URL is not reachable within the probe timeout."""

NO_TASK_ASSIGNED       = "NO_TASK_ASSIGNED"
"""No enabled tasks are configured for the given camera channel."""

TASK_NOT_FOUND         = "TASK_NOT_FOUND"
"""A referenced task ID does not exist in the task registry."""

TASK_DISABLED          = "TASK_DISABLED"
"""Task exists but its ``enable`` flag is False."""

CAMERA_TASK_MISMATCH   = "CAMERA_TASK_MISMATCH"
"""Task's channelId does not match the camera being started."""

TASK_CONFIG_INVALID    = "TASK_CONFIG_INVALID"
"""Task configuration fails schema or geometry validation."""

# ── Init-phase codes (subprocess startup) ────────────────────────────────────

MODEL_INIT_FAILED      = "MODEL_INIT_FAILED"
"""YOLO / AI model failed to initialize inside the FrameBus subprocess."""

WORKER_INIT_FAILED     = "WORKER_INIT_FAILED"
"""Task worker failed to initialize its algorithm inside the worker subprocess."""

WORKER_INIT_TIMEOUT    = "WORKER_INIT_TIMEOUT"
"""Task worker did not signal ready within the configured timeout."""

BUS_INIT_TIMEOUT       = "BUS_INIT_TIMEOUT"
"""FrameBus subprocess did not signal ready within the configured timeout."""

# ── Runtime codes (emitted as SSE events during active stream) ────────────────

TASK_REMOVED_RUNTIME   = "TASK_REMOVED_RUNTIME"
"""Task was deleted from the registry while the worker was processing frames."""

TASK_DISABLED_RUNTIME  = "TASK_DISABLED_RUNTIME"
"""Task was disabled (enable=false) while the worker was processing frames."""

STREAM_DISCONNECTED    = "STREAM_DISCONNECTED"
"""RTSP/video stream disconnected or became unreachable during processing."""

ANNOTATION_STOPPED     = "ANNOTATION_STOPPED"
"""Annotation was stopped due to a validation or runtime failure."""

WORKER_CRASHED         = "WORKER_CRASHED"
"""A task worker process crashed unexpectedly."""

WORKER_ERROR           = "WORKER_ERROR"
"""A task worker encountered an error processing a frame (recoverable)."""


# ── Structured error types ────────────────────────────────────────────────────

@dataclass
class ValidationFailure:
    """
    Returned by ``StreamValidator`` when any pre-spawn validation step fails.

    Serialises to a JSON-safe dict via ``to_dict()``.  The ``stage`` field
    (1–10) maps directly to the 10-step validation sequence documented in the
    plan; ``stage_name`` is the human-readable equivalent.
    """
    stage      : int
    stage_name : str
    error_code : str
    message    : str
    stream_id  : str
    camera_id  : Optional[str] = None
    task_id    : Optional[str] = None
    timestamp  : float = field(default_factory=time.time)
    details    : Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["timestamp_utc"] = _fmt_ts(self.timestamp)
        return d

    def user_message(self) -> str:
        return (
            f"Annotation blocked [{self.error_code}] at stage {self.stage} "
            f"({self.stage_name}): {self.message}"
        )


@dataclass
class RuntimeError_:
    """
    Structured error emitted on the SSE/result-queue path when a worker
    encounters a failure during active stream processing.  Field names match
    the existing event envelope so clients can parse them uniformly.
    """
    error_code  : str
    message     : str
    camera_id   : str
    task_id     : Optional[str] = None
    stage       : Optional[str] = None
    timestamp   : float = field(default_factory=time.time)
    details     : Dict[str, Any] = field(default_factory=dict)

    def to_event_dict(self) -> dict:
        return {
            "eventType"    : "PIPELINE_ERROR",
            "error_code"   : self.error_code,
            "message"      : self.message,
            "camera_id"    : self.camera_id,
            "task_id"      : self.task_id,
            "stage"        : self.stage,
            "timestamp"    : int(self.timestamp * 1000),
            "timestamp_utc": _fmt_ts(self.timestamp),
            "details"      : self.details,
        }


def _fmt_ts(ts: float) -> str:
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")
