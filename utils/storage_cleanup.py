"""
TTL-based cleanup for pipeline output and evidence directories.

Controlled by OUTPUT_RETENTION_HOURS (default 24). Set to 0 to disable sweeps.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Iterable

log = logging.getLogger(__name__)


def _retention_seconds() -> float:
    try:
        hours = float(os.getenv("OUTPUT_RETENTION_HOURS", "24"))
    except ValueError:
        hours = 24.0
    if hours <= 0:
        return 0.0
    return hours * 3600.0


def _cleanup_roots() -> list[str]:
    roots = [
        os.getenv("OUTPUT_DIR", "./outputs"),
        os.getenv("CAPTURE_DIR", "./evidence/capture"),
        os.getenv("SCENE_DIR", "./evidence/scene"),
        os.getenv("GALLERY_DIR", "/local/storage/gallery"),
        os.getenv("CASHIER_EVIDENCE_DIR", "./evidence/cashier"),
        os.getenv("CAMERA_SNAPSHOT_DIR", "./outputs/camera_snapshots"),
    ]
    out: list[str] = []
    for r in roots:
        if r and r not in out:
            out.append(r)
    return out


def _iter_files(root: str) -> Iterable[str]:
    if not root or not os.path.isdir(root):
        return
    try:
        for dirpath, _dirnames, filenames in os.walk(root):
            for fn in filenames:
                yield os.path.join(dirpath, fn)
    except OSError as exc:
        log.debug("storage_cleanup walk failed for %s: %s", root, exc)


def sweep_old_files(max_age_sec: float) -> tuple[int, int]:
    """
    Delete regular files under configured roots older than max_age_sec.
    Returns (files_deleted, bytes_freed) best-effort.
    """
    if max_age_sec <= 0:
        return 0, 0
    now = time.time()
    deleted = 0
    freed = 0
    for root in _cleanup_roots():
        for path in _iter_files(root):
            try:
                st = os.stat(path)
            except OSError:
                continue
            if not os.path.isfile(path):
                continue
            if now - st.st_mtime <= max_age_sec:
                continue
            try:
                freed += st.st_size
                os.remove(path)
                deleted += 1
            except OSError as exc:
                log.debug("storage_cleanup could not remove %s: %s", path, exc)
    return deleted, freed


def _sweep_interval_sec() -> float:
    try:
        return max(60.0, float(os.getenv("STORAGE_CLEANUP_INTERVAL_SEC", "3600")))
    except ValueError:
        return 3600.0


def start_storage_cleanup_thread() -> threading.Thread | None:
    """
    Start a daemon thread that periodically deletes files older than OUTPUT_RETENTION_HOURS.
    Returns None if retention is disabled (OUTPUT_RETENTION_HOURS <= 0).
    """
    max_age = _retention_seconds()
    if max_age <= 0:
        log.info(
            "storage_cleanup disabled (OUTPUT_RETENTION_HOURS=%r)",
            os.getenv("OUTPUT_RETENTION_HOURS", "24"),
        )
        return None

    interval = _sweep_interval_sec()
    stop = threading.Event()

    def _run() -> None:
        log.info(
            "storage_cleanup started: max_age=%.0fs interval=%.0fs roots=%s",
            max_age,
            interval,
            _cleanup_roots(),
        )
        while not stop.wait(timeout=interval):
            try:
                n, b = sweep_old_files(max_age)
                if n:
                    log.info(
                        "storage_cleanup removed %d files (~%d bytes)", n, b
                    )
            except Exception:
                log.exception("storage_cleanup sweep failed")

    t = threading.Thread(target=_run, name="storage_cleanup", daemon=True)
    t.start()
    return t
