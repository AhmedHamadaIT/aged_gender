"""
tests/unit/test_multi_camera_throughput.py
-------------------------------------------
S-8: Multi-camera concurrent throughput test (pure Python, no YOLO model).

Simulates N cameras simultaneously submitting frames to a shared task registry
and verifies that:
  1. Each camera's tasks are processed independently.
  2. No frames are lost under concurrent load.
  3. Per-task confThreshold filtering is applied correctly.
"""

from __future__ import annotations

import concurrent.futures
import queue
import time

import pytest


def _make_task(task_id: int, camera_id: str, conf_threshold: float = 0.0) -> dict:
    return {
        "taskId": task_id,
        "taskName": f"task_{task_id}",
        "algorithmType": "CROSS_LINE",
        "channelId": camera_id,
        "enable": True,
        "detailConfig": {"confThreshold": conf_threshold},
    }


def _make_detection(conf: float) -> dict:
    return {
        "x1": 10, "y1": 10, "x2": 50, "y2": 80,
        "class_id": 0, "class_name": "person",
        "confidence": conf, "track_id": 1,
    }


def _make_payload(camera_id: str, conf: float) -> dict:
    return {
        "camera_id": camera_id,
        "timestamp": time.time(),
        "detection": {
            "items": [_make_detection(conf)],
            "count": 1,
        },
    }


def _conf_filter(payload: dict, threshold: float) -> dict:
    """Mirror of the M-4 confThreshold logic in task_worker."""
    if threshold <= 0.0:
        return payload
    det = payload.get("detection") or {}
    items = det.get("items") or []
    filtered = [d for d in items if float(d.get("confidence", 1.0)) >= threshold]
    if len(filtered) == len(items):
        return payload
    return {**payload, "detection": {**det, "items": filtered, "count": len(filtered)}}


def _worker(camera_id: str, task: dict, payloads: list, result_q: queue.Queue):
    threshold = float((task.get("detailConfig") or {}).get("confThreshold", 0.0))
    passed = 0
    for p in payloads:
        filtered = _conf_filter(p, threshold)
        if filtered["detection"]["count"] > 0:
            passed += 1
    result_q.put((camera_id, passed))


@pytest.mark.parametrize("n_cameras,frames_per_camera,conf_threshold", [
    (4, 500, 0.0),
    (4, 500, 0.5),
    (8, 200, 0.25),
])
def test_multi_camera_no_frame_loss(n_cameras, frames_per_camera, conf_threshold):
    """
    All frames above the threshold must be counted, across N concurrent camera threads.
    """
    cameras = [f"cam_{i}" for i in range(n_cameras)]
    tasks = {cam: _make_task(i, cam, conf_threshold) for i, cam in enumerate(cameras)}

    # Payloads alternating between conf=0.8 (pass) and conf=0.3 (may fail threshold).
    payloads_map = {}
    expected_map = {}
    for cam in cameras:
        payloads = []
        expected = 0
        for j in range(frames_per_camera):
            conf = 0.8 if j % 2 == 0 else 0.3
            payloads.append(_make_payload(cam, conf))
            if conf >= conf_threshold:
                expected += 1
        payloads_map[cam] = payloads
        expected_map[cam] = expected

    result_q: queue.Queue = queue.Queue()

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_cameras) as executor:
        futs = [
            executor.submit(_worker, cam, tasks[cam], payloads_map[cam], result_q)
            for cam in cameras
        ]
        concurrent.futures.wait(futs)

    results = {}
    while not result_q.empty():
        cam, passed = result_q.get_nowait()
        results[cam] = passed

    assert len(results) == n_cameras, "Some camera workers did not report results"
    for cam in cameras:
        assert results[cam] == expected_map[cam], (
            f"[{cam}] expected {expected_map[cam]} frames, got {results[cam]}"
        )
