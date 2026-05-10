"""Bounded in-memory event buffer with optional SQLite spill."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from collections import deque
from typing import Any, Deque, List, Optional, Tuple


class EventBuffer:
    """
    FIFO buffer for JSON-serializable dicts. When in-memory max is reached,
    optionally append to SQLite WAL (if db_path set).
    """

    def __init__(
        self,
        max_memory: int = 1000,
        db_path: Optional[str] = None,
    ) -> None:
        self._max_memory = max(1, int(max_memory))
        self._db_path = db_path
        self._deque: Deque[str] = deque()
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        if self._db_path:
            _dir = os.path.dirname(os.path.abspath(self._db_path))
            if _dir:
                os.makedirs(_dir, exist_ok=True)
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS spill (id INTEGER PRIMARY KEY AUTOINCREMENT, payload TEXT NOT NULL)"
            )
            self._conn.commit()

    def __len__(self) -> int:
        with self._lock:
            n = len(self._deque)
            if self._conn:
                cur = self._conn.execute("SELECT COUNT(*) FROM spill")
                n += int(cur.fetchone()[0])
            return n

    def append(self, event: dict) -> Tuple[bool, str]:
        """
        Try to store event. Returns (ok, reason).
        If memory full and spill fails or disabled, returns (False, "dropped").
        """
        line = json.dumps(event, separators=(",", ":"))
        with self._lock:
            if len(self._deque) < self._max_memory:
                self._deque.append(line)
                return True, "memory"
            if self._conn is not None:
                try:
                    self._conn.execute("INSERT INTO spill (payload) VALUES (?)", (line,))
                    self._conn.commit()
                    return True, "disk"
                except Exception:
                    return False, "spill_failed"
            return False, "full"

    def popleft_batch(self, limit: int) -> List[dict]:
        """Pop up to `limit` events (memory first, then disk)."""
        out: List[dict] = []
        with self._lock:
            while len(out) < limit and self._deque:
                line = self._deque.popleft()
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            while len(out) < limit and self._conn is not None:
                cur = self._conn.execute("SELECT id, payload FROM spill ORDER BY id LIMIT 1")
                row = cur.fetchone()
                if not row:
                    break
                rid, payload = row
                self._conn.execute("DELETE FROM spill WHERE id = ?", (rid,))
                self._conn.commit()
                try:
                    out.append(json.loads(payload))
                except json.JSONDecodeError:
                    continue
        return out

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None
