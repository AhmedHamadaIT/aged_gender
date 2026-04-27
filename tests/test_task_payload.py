"""Tests for utils.task_payload — FrameBus → task worker frame materialization."""

from __future__ import annotations

import base64
import os

import cv2
import numpy as np
import pytest


def test_task_frame_bgr_decodes_jpeg(monkeypatch):
    monkeypatch.delenv("TASK_QUEUE_INCLUDE_FRAME", raising=False)
    img = np.zeros((24, 32, 3), dtype=np.uint8)
    img[:, :, 1] = 200
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    assert ok
    from utils.task_payload import task_frame_bgr

    out = task_frame_bgr({"frame_b64": base64.b64encode(bytes(buf)).decode("ascii")})
    assert out.shape[:2] == (24, 32)
    assert out.dtype == np.uint8


def test_task_frame_bgr_prefers_ndarray_when_include_frame(monkeypatch):
    monkeypatch.setenv("TASK_QUEUE_INCLUDE_FRAME", "true")
    arr = np.zeros((10, 10, 3), dtype=np.uint8)
    arr[3, 3] = (1, 2, 3)
    from utils.task_payload import task_frame_bgr

    out = task_frame_bgr({"frame": arr, "frame_b64": ""})
    assert out[3, 3].tolist() == [1, 2, 3]


def test_task_frame_bgr_missing_raises():
    from utils.task_payload import task_frame_bgr

    with pytest.raises(ValueError, match="frame_b64"):
        task_frame_bgr({})
