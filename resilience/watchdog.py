"""Async watchdog: poll FrameBus / task worker processes and trigger respawn."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

PollFn = Callable[[], None]


def _env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).lower() in ("true", "1", "yes", "on")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return int(default)


class DetectionWatchdog:
    """
    Periodically checks registered processes; invokes respawn callback when dead.
    Respawn budget: WATCHDOG_MAX_RESPAWNS per WATCHDOG_RESPAWN_WINDOW_SEC (per camera).
    """

    def __init__(
        self,
        *,
        poll_interval_sec: Optional[float] = None,
        respawn_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._poll_interval = poll_interval_sec
        if self._poll_interval is None:
            self._poll_interval = _env_float("WATCHDOG_POLL_INTERVAL_SEC", 2.0)
        self._respawn_callback = respawn_callback
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._respawn_times: dict[str, list[float]] = {}

    async def start(self) -> None:
        if self._running:
            return
        if not _env_bool("WATCHDOG_ENABLED", "false"):
            log.info("detection_watchdog disabled (WATCHDOG_ENABLED=false)")
            return
        self._running = True
        self._task = asyncio.create_task(self._loop(), name="detection_watchdog")

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    def _within_budget(self, cam_id: str) -> bool:
        window = _env_float("WATCHDOG_RESPAWN_WINDOW_SEC", 600.0)
        cap = _env_int("WATCHDOG_MAX_RESPAWNS", 5)
        now = time.time()
        times = [t for t in self._respawn_times.get(cam_id, []) if now - t <= window]
        self._respawn_times[cam_id] = times
        return len(times) < cap

    def _record_respawn(self, cam_id: str) -> None:
        self._respawn_times.setdefault(cam_id, []).append(time.time())

    async def _loop(self) -> None:
        while self._running:
            try:
                if self._respawn_callback:
                    self._respawn_callback()
            except Exception:
                log.exception("detection_watchdog poll callback failed")
            try:
                await asyncio.sleep(self._poll_interval)
            except asyncio.CancelledError:
                break


def schedule_watchdog(
    *,
    get_detection: Callable[[], Any],
    loop: asyncio.AbstractEventLoop,
) -> DetectionWatchdog:
    """
    Build a watchdog whose callback checks detection's processes and calls
    detection._watchdog_tick() if present.
    """

    def _tick() -> None:
        det = get_detection()
        if det is None:
            return
        fn = getattr(det, "_watchdog_tick", None)
        if callable(fn):
            fn()

    wd = DetectionWatchdog(respawn_callback=_tick)
    return wd
