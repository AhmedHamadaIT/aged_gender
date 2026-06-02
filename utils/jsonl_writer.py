"""
utils/jsonl_writer.py
---------------------
M-11: Shared JSONL append helper with optional daily rotation and retention.

Environment variables:
    JSONL_ROTATE         bool (default "false") — enable daily rotation.
    JSONL_RETAIN_DAYS    int  (default 30)      — days of rotated files to keep.
                                                  0 = keep forever.

Usage:
    from utils.jsonl_writer import JsonlWriter

    writer = JsonlWriter(Path("/output/my_dir/events.jsonl"))
    writer.append({"key": "value", "timestamp": "..."})

When JSONL_ROTATE is off (default), this is a pure passthrough to the existing
open(..., "a") pattern — no behaviour change for existing code.

When JSONL_ROTATE is on, the writer creates dated shard files:
    events_2026-05-23.jsonl  ← today's active shard
    events_2026-05-22.jsonl  ← yesterday's shard (read-only)
    ...

Old shards beyond JSONL_RETAIN_DAYS are deleted on append (lazy GC).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

_ROTATE: bool = os.getenv("JSONL_ROTATE", "false").lower() in ("true", "1", "yes")
_RETAIN_DAYS: int = max(0, int(os.getenv("JSONL_RETAIN_DAYS", "30")))


class JsonlWriter:
    """
    Thread-compatible JSONL writer.

    The writer is NOT thread-safe by itself — callers that share a writer
    across threads should hold their own lock (same pattern as existing code).
    """

    def __init__(self, path: Path) -> None:
        """
        ``path`` is the canonical base path (e.g. ``/output/cam1/events.jsonl``).
        The stem and suffix are used when rotation is on.
        """
        self._base = path
        self._rotate = _ROTATE
        self._retain = _RETAIN_DAYS

    def _active_path(self) -> Path:
        """Return the path that should be written to right now."""
        if not self._rotate:
            return self._base
        today = date.today().isoformat()
        return self._base.with_name(f"{self._base.stem}_{today}{self._base.suffix}")

    def append(self, record: dict) -> None:
        """Append one JSON record followed by a newline to the active shard."""
        target = self._active_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False)
        with open(target, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        if self._rotate and self._retain > 0:
            self._gc()

    def _gc(self) -> None:
        """Delete shard files older than JSONL_RETAIN_DAYS."""
        cutoff = date.today() - timedelta(days=self._retain)
        stem = self._base.stem
        suffix = self._base.suffix
        parent = self._base.parent
        if not parent.exists():
            return
        for p in parent.glob(f"{stem}_*{suffix}"):
            try:
                date_part = p.stem[len(stem) + 1:]  # strip "<stem>_"
                file_date = date.fromisoformat(date_part)
                if file_date < cutoff:
                    p.unlink(missing_ok=True)
            except (ValueError, OSError):
                pass


def open_jsonl(path: Path) -> "JsonlWriter":
    """Convenience factory — returns a JsonlWriter for the given path."""
    return JsonlWriter(path)
