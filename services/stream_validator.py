"""
services/stream_validator.py
-----------------------------
Strict 10-step validation chain that MUST pass before any FrameBus or task
worker process is spawned.

Steps 1–7 run synchronously in the API process (no subprocesses created).
Steps 8–9 are handled externally via init-events (see apis/detection.py).
Step 10 is the implicit result of all previous steps passing.

Call:
    validator = StreamValidator(camera_registry, task_registry)
    failure = validator.validate_channel(cam_id, rtsp_url, chan_tasks)
    if failure:
        # Return 422 with failure.to_dict() — no processes spawned

Design contract:
- Returns the FIRST failure encountered; never continues past a failure.
- Never spawns subprocesses, opens model files, or performs slow I/O
  except the lightweight stream-reachability probe (step 2).
- All check methods are pure and individually testable.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import time
from typing import Dict, List, Optional, Any
from urllib.parse import urlparse

from utils.error_codes import (
    ValidationFailure,
    STREAM_NOT_REGISTERED,
    STREAM_UNREACHABLE,
    NO_TASK_ASSIGNED,
    TASK_NOT_FOUND,
    TASK_DISABLED,
    CAMERA_TASK_MISMATCH,
    TASK_CONFIG_INVALID,
)

log = logging.getLogger(__name__)

# Default probe timeout (seconds) — env-overridable without restarting the service.
_DEFAULT_PROBE_TIMEOUT = float(os.getenv("STREAM_PROBE_TIMEOUT_SEC", "3.0"))
# Set STREAM_PROBE_ENABLED=false to skip the network probe (useful for local file streams
# or when the network topology prevents a pre-flight connect from the API process).
_PROBE_ENABLED = os.getenv("STREAM_PROBE_ENABLED", "true").lower() not in ("false", "0", "no")


class StreamValidator:
    """
    Stateless validator: each method takes explicit arguments so it can be
    unit-tested in isolation without a full app context.

    ``validate_channel`` runs all steps in strict order and returns the first
    ``ValidationFailure`` (or ``None`` when everything passes).
    """

    def __init__(self, camera_registry, task_registry):
        """
        Args:
            camera_registry: Object exposing ``.all() -> Dict[str, str]``
                             (camera_id → rtsp_url).
            task_registry:   Object exposing ``.get(task_id: int) -> Optional[dict]``
                             and ``.get_enabled() -> List[dict]``.
        """
        self._cameras = camera_registry
        self._tasks = task_registry

    # ── Public entry point ────────────────────────────────────────────────────

    def validate_channel(
        self,
        cam_id: str,
        rtsp_url: str,
        chan_tasks: List[dict],
    ) -> Optional[ValidationFailure]:
        """
        Run all 7 in-process validation steps for *cam_id*.

        Returns the first ``ValidationFailure`` encountered, or ``None`` if all
        steps pass.  The caller MUST NOT spawn any processes when a failure is
        returned.

        Steps performed:
          1. Camera ID exists in registry
          2. Stream URL is reachable (socket probe, skippable via env)
          3. At least one task is assigned to the channel
          4. Each task ID exists in the registry (not just in-memory list)
          5. Each task has ``enable=True``
          6. Each task's ``channelId`` matches ``cam_id``
          7. Each task's configuration is structurally valid
        """
        ctx = _Ctx(cam_id=cam_id, rtsp_url=rtsp_url)

        failure = self._check_stream_registered(ctx)
        if failure:
            return failure

        failure = self._check_stream_reachable(ctx)
        if failure:
            return failure

        failure = self._check_tasks_assigned(ctx, chan_tasks)
        if failure:
            return failure

        for task in chan_tasks:
            failure = self._check_task_exists(ctx, task)
            if failure:
                return failure

            failure = self._check_task_enabled(ctx, task)
            if failure:
                return failure

            failure = self._check_task_camera_mapping(ctx, task)
            if failure:
                return failure

            failure = self._check_task_config_valid(ctx, task)
            if failure:
                return failure

        log.info(
            "[StreamValidator] cam=%s passed all 7 pre-spawn checks (%d task(s))",
            cam_id, len(chan_tasks),
        )
        return None

    # ── Step 1: stream registered ─────────────────────────────────────────────

    def _check_stream_registered(self, ctx: "_Ctx") -> Optional[ValidationFailure]:
        cameras = self._cameras.all()
        if ctx.cam_id not in cameras:
            return ValidationFailure(
                stage=1,
                stage_name="stream_registered",
                error_code=STREAM_NOT_REGISTERED,
                message=(
                    f"Annotation blocked: camera '{ctx.cam_id}' is not registered. "
                    f"Call POST /cameras with id='{ctx.cam_id}' and a valid RTSP URL."
                ),
                stream_id=ctx.cam_id,
                camera_id=ctx.cam_id,
                details={"registered_cameras": sorted(cameras.keys())},
            )
        return None

    # ── Step 2: stream reachable ──────────────────────────────────────────────

    def _check_stream_reachable(self, ctx: "_Ctx") -> Optional[ValidationFailure]:
        if not _PROBE_ENABLED:
            log.debug(
                "[StreamValidator] cam=%s stream probe disabled (STREAM_PROBE_ENABLED=false)",
                ctx.cam_id,
            )
            return None

        reachable, probe_detail = _probe_stream_url(ctx.rtsp_url, _DEFAULT_PROBE_TIMEOUT)
        if not reachable:
            return ValidationFailure(
                stage=2,
                stage_name="stream_reachable",
                error_code=STREAM_UNREACHABLE,
                message=(
                    f"Annotation blocked: stream URL for camera '{ctx.cam_id}' is not "
                    f"reachable within {_DEFAULT_PROBE_TIMEOUT:.1f}s. "
                    f"Check network connectivity and the RTSP URL."
                ),
                stream_id=ctx.cam_id,
                camera_id=ctx.cam_id,
                details={"rtsp_url": ctx.rtsp_url, "probe_detail": probe_detail,
                         "probe_timeout_sec": _DEFAULT_PROBE_TIMEOUT},
            )
        return None

    # ── Step 3: tasks assigned ────────────────────────────────────────────────

    def _check_tasks_assigned(
        self, ctx: "_Ctx", chan_tasks: List[dict]
    ) -> Optional[ValidationFailure]:
        if not chan_tasks:
            all_enabled = self._tasks.get_enabled()
            assigned_channels = sorted({str(t.get("channelId", "")) for t in all_enabled})
            return ValidationFailure(
                stage=3,
                stage_name="tasks_assigned",
                error_code=NO_TASK_ASSIGNED,
                message=(
                    f"Annotation blocked: no tasks assigned to stream '{ctx.cam_id}'. "
                    f"Create a task with channelId='{ctx.cam_id}' via POST /api/tasks."
                ),
                stream_id=ctx.cam_id,
                camera_id=ctx.cam_id,
                details={"channels_with_tasks": assigned_channels},
            )
        return None

    # ── Step 4: task exists in registry ──────────────────────────────────────

    def _check_task_exists(
        self, ctx: "_Ctx", task: dict
    ) -> Optional[ValidationFailure]:
        task_id = task.get("taskId")
        if task_id is None or self._tasks.get(int(task_id)) is None:
            return ValidationFailure(
                stage=4,
                stage_name="task_exists",
                error_code=TASK_NOT_FOUND,
                message=(
                    f"Annotation blocked: task ID '{task_id}' not found in registry "
                    f"for camera '{ctx.cam_id}'. "
                    f"Ensure the task was registered via POST /api/tasks."
                ),
                stream_id=ctx.cam_id,
                camera_id=ctx.cam_id,
                task_id=str(task_id),
            )
        return None

    # ── Step 5: task enabled ──────────────────────────────────────────────────

    def _check_task_enabled(
        self, ctx: "_Ctx", task: dict
    ) -> Optional[ValidationFailure]:
        task_id = str(task.get("taskId", "?"))
        if not task.get("enable", True):
            return ValidationFailure(
                stage=5,
                stage_name="task_enabled",
                error_code=TASK_DISABLED,
                message=(
                    f"Annotation blocked: task '{task_id}' ('{task.get('taskName', '')}') "
                    f"is disabled (enable=false). "
                    f"Enable the task via PUT /api/tasks/{task_id} with enable=true."
                ),
                stream_id=ctx.cam_id,
                camera_id=ctx.cam_id,
                task_id=task_id,
                details={"task_name": task.get("taskName"), "algorithm": task.get("algorithmType")},
            )
        return None

    # ── Step 6: camera-task mapping ───────────────────────────────────────────

    def _check_task_camera_mapping(
        self, ctx: "_Ctx", task: dict
    ) -> Optional[ValidationFailure]:
        task_id = str(task.get("taskId", "?"))
        task_channel = str(task.get("channelId", ""))
        if task_channel != ctx.cam_id:
            return ValidationFailure(
                stage=6,
                stage_name="camera_task_mapping",
                error_code=CAMERA_TASK_MISMATCH,
                message=(
                    f"Annotation blocked: invalid camera-task mapping for task '{task_id}'. "
                    f"Task channelId='{task_channel}' does not match camera '{ctx.cam_id}'. "
                    f"Update the task's channelId or start the correct camera."
                ),
                stream_id=ctx.cam_id,
                camera_id=ctx.cam_id,
                task_id=task_id,
                details={"task_channel_id": task_channel, "started_camera_id": ctx.cam_id},
            )
        return None

    # ── Step 7: task config valid ─────────────────────────────────────────────

    def _check_task_config_valid(
        self, ctx: "_Ctx", task: dict
    ) -> Optional[ValidationFailure]:
        task_id = str(task.get("taskId", "?"))
        algorithm = task.get("algorithmType", "")

        error = _validate_task_config(task)
        if error:
            return ValidationFailure(
                stage=7,
                stage_name="task_config_valid",
                error_code=TASK_CONFIG_INVALID,
                message=(
                    f"Annotation blocked: invalid configuration for task '{task_id}' "
                    f"({algorithm}). {error}"
                ),
                stream_id=ctx.cam_id,
                camera_id=ctx.cam_id,
                task_id=task_id,
                details={"algorithm": algorithm, "config_error": error},
            )
        return None


# ── Helpers ───────────────────────────────────────────────────────────────────

class _Ctx:
    """Lightweight value object — avoids repeating positional args."""
    __slots__ = ("cam_id", "rtsp_url")

    def __init__(self, cam_id: str, rtsp_url: str):
        self.cam_id  = cam_id
        self.rtsp_url = rtsp_url


def _probe_stream_url(url: str, timeout_sec: float) -> tuple[bool, str]:
    """
    Lightweight connectivity probe.

    - Local file paths / ``file://`` URIs: check existence only.
    - RTSP / HTTP(S): attempt TCP socket connection to host:port.
    - Returns ``(reachable: bool, detail: str)``.
    """
    if not url:
        return False, "empty URL"

    # Local file
    if url.startswith("file://"):
        path = url[7:]
        exists = os.path.isfile(path)
        return exists, f"file {'exists' if exists else 'not found'}: {path}"
    if not url.startswith(("rtsp://", "rtsps://", "http://", "https://", "rtmp://")):
        # Treat as local file path
        exists = os.path.isfile(url)
        return exists, f"file {'exists' if exists else 'not found'}: {url}"

    try:
        parsed = urlparse(url)
        host = parsed.hostname
        if not host:
            return False, "could not parse hostname from URL"
        scheme = parsed.scheme.lower()
        default_ports = {"rtsp": 554, "rtsps": 322, "http": 80, "https": 443, "rtmp": 1935}
        port = parsed.port or default_ports.get(scheme, 554)

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout_sec)
        try:
            sock.connect((host, port))
            return True, f"TCP connect ok to {host}:{port}"
        except socket.timeout:
            return False, f"TCP connect timed out after {timeout_sec:.1f}s to {host}:{port}"
        except ConnectionRefusedError:
            return False, f"TCP connection refused at {host}:{port}"
        except OSError as exc:
            return False, f"TCP connect error to {host}:{port}: {exc}"
        finally:
            try:
                sock.close()
            except Exception:
                pass
    except Exception as exc:
        return False, f"probe error: {exc}"


def _validate_task_config(task: dict) -> Optional[str]:
    """
    Return an error string if the task config is invalid, else None.

    Currently validates:
    - CROSS_LINE: areaPosition must parse to ≥1 valid line.
    - All tasks: threshold must be 0-100.
    - All tasks: algorithmType must be a non-empty string.
    """
    algorithm = task.get("algorithmType", "")
    if not algorithm or not isinstance(algorithm, str):
        return "algorithmType is missing or empty"

    threshold = task.get("threshold", 50)
    try:
        t = int(threshold)
        if not (0 <= t <= 100):
            return f"threshold must be 0-100, got {t}"
    except (TypeError, ValueError):
        return f"threshold must be an integer, got {threshold!r}"

    if algorithm == "CROSS_LINE":
        area_position = task.get("areaPosition", "[]")
        if not area_position or not str(area_position).strip() or str(area_position).strip() == "[]":
            return (
                "CROSS_LINE task requires a non-empty areaPosition with at least one valid line."
            )
        try:
            parsed = json.loads(str(area_position))
        except (json.JSONDecodeError, TypeError) as exc:
            return f"areaPosition is not valid JSON: {exc}"
        if not isinstance(parsed, list) or not parsed:
            return "areaPosition must be a JSON array with at least one line object."
        # Defer to parse_effective_cross_lines for detailed geometry validation
        try:
            from services.cross_line import parse_effective_cross_lines
            lines = parse_effective_cross_lines(area_position)
            if not lines:
                return (
                    "areaPosition contains no valid lines. Each line needs "
                    "\"point\" with two {x,y} objects with numeric coordinates."
                )
        except Exception as exc:
            return f"areaPosition geometry validation error: {exc}"

    return None
