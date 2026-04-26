"""
Task worker payload helpers.

FrameBus sends ``frame_b64`` (JPEG) to task processes. Including a raw BGR
``numpy`` array in every ``multiprocessing.Queue`` message forces large pickles
over the pipe and commonly saturates ``TASK_QUEUE_MAXSIZE``.

Default: workers decode ``frame_b64`` when they need pixels. Set
``TASK_QUEUE_INCLUDE_FRAME=true`` to restore the legacy ``payload['frame']``
ndarray (higher IPC cost).
"""

from __future__ import annotations

import base64
import os
from typing import Any

import cv2
import numpy as np


def task_frame_bgr(payload: dict[str, Any]) -> np.ndarray:
    """
    Return a BGR ``uint8`` image for task algorithms.

    Uses ``payload['frame']`` when present and ``TASK_QUEUE_INCLUDE_FRAME`` is
    enabled; otherwise decodes ``payload['frame_b64']``.
    """
    include = os.getenv("TASK_QUEUE_INCLUDE_FRAME", "false").lower() in (
        "true",
        "1",
        "yes",
    )
    if include:
        f = payload.get("frame")
        if f is not None:
            return f

    b64 = payload.get("frame_b64")
    if not b64:
        raise ValueError("task payload missing frame_b64 (and no frame array)")

    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("task payload: JPEG imdecode failed")
    return img
