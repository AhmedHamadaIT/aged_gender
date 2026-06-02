"""
utils/shm_ring.py
-----------------
M-2: Optional shared-memory ring buffer for task fan-out.

When TASK_SHM_ENABLED=true, FrameBus writes each JPEG into a fixed-size
shared-memory block and enqueues only a lightweight FrameRef (name + slice)
to each task queue.  Task workers read bytes directly from SHM, eliminating
N-1 extra pickle copies (one SHM write vs. N queue payloads).

The ring has TASK_SHM_RING slots of TASK_SHM_SLOT_KB KB each.  When all
slots are occupied (very slow consumers) the producer falls back to the
normal pickle path for that frame.

Default: TASK_SHM_ENABLED=false → this module is imported but not activated.
"""

from __future__ import annotations

import os
import struct
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class FrameRef:
    """Lightweight token enqueued to task queues instead of the full JPEG bytes."""
    shm_name: str
    offset: int
    size: int
    frame_id: int
    camera_id: str
    timestamp: str
    detections: list
    # Detection count (convenience)
    count: int


_SHM_ENABLED: bool = os.getenv("TASK_SHM_ENABLED", "false").lower() in ("true", "1", "yes")
_SHM_RING_SIZE: int = max(2, int(os.getenv("TASK_SHM_RING", "8")))
_SHM_SLOT_KB: int = max(64, int(os.getenv("TASK_SHM_SLOT_KB", "512")))
_SHM_SLOT_BYTES: int = _SHM_SLOT_KB * 1024


class ShmFrameRing:
    """
    Fixed-size ring of shared-memory slots.  Thread-safe for a single producer.
    """

    def __init__(self, name_prefix: str, ring_size: int = _SHM_RING_SIZE, slot_bytes: int = _SHM_SLOT_BYTES):
        from multiprocessing.shared_memory import SharedMemory

        self._slot_bytes = slot_bytes
        self._ring_size = ring_size
        self._slots: list[SharedMemory] = []
        self._lock = threading.Lock()
        self._cursor = 0

        for i in range(ring_size):
            shm = SharedMemory(
                name=f"{name_prefix}_{i}",
                create=True,
                size=slot_bytes + 8,  # 8 bytes header: uint64 size
            )
            self._slots.append(shm)

    def write(self, jpeg_bytes: bytes) -> Optional[tuple[str, int, int]]:
        """
        Write JPEG bytes into the next ring slot.
        Returns (shm_name, offset, size) or None if data is too large.
        """
        size = len(jpeg_bytes)
        if size > self._slot_bytes:
            return None

        with self._lock:
            idx = self._cursor % self._ring_size
            self._cursor += 1

        slot = self._slots[idx]
        # Write size header then JPEG bytes.
        struct.pack_into("<Q", slot.buf, 0, size)
        slot.buf[8: 8 + size] = jpeg_bytes
        return slot.name, 8, size

    def close(self) -> None:
        for shm in self._slots:
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass
        self._slots.clear()


def read_frame_ref(ref: FrameRef) -> bytes:
    """
    Read JPEG bytes from shared memory given a FrameRef.
    Falls back gracefully when the SHM segment has been recycled.
    """
    try:
        from multiprocessing.shared_memory import SharedMemory

        shm = SharedMemory(name=ref.shm_name, create=False)
        data = bytes(shm.buf[ref.offset: ref.offset + ref.size])
        shm.close()
        return data
    except Exception:
        return b""


def frame_ref_to_payload(ref: FrameRef) -> dict:
    """Convert a FrameRef back into a payload dict compatible with task_frame_bgr."""
    import base64

    jpeg = read_frame_ref(ref)
    b64 = base64.b64encode(jpeg).decode("utf-8") if jpeg else ""
    return {
        "camera_id": ref.camera_id,
        "frame_id": ref.frame_id,
        "timestamp": ref.timestamp,
        "frame_b64": b64,
        "detection": {
            "items": ref.detections,
            "count": ref.count,
        },
    }
