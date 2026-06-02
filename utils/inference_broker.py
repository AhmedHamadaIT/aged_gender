"""
utils/inference_broker.py
--------------------------
S-3: Optional batched inference broker process.

When INFERENCE_BROKER_ENABLED=true, a separate process owns the YOLO model
and accepts (camera_id, frame_ndarray) via multiprocessing.Queue, runs
model.track() in batches (INFERENCE_BATCH_SIZE), and returns results via
per-camera reply queues.

This is a **stub/framework** for the broker pattern.  Full batching requires
restructuring FrameBus so that each camera process submits frames to the broker
queue and awaits results instead of calling model.track() directly — a major
refactor that is gated behind INFERENCE_BROKER_ENABLED to keep the default code
path completely unchanged.

The implementation here provides:
 - InferenceBroker class with start/stop lifecycle.
 - InferenceRequest / InferenceResult dataclasses.
 - run_inference_broker() entry point (for multiprocessing.Process).
 - is_broker_enabled() helper used by FrameBus to decide whether to submit
   frames to the broker or run inference locally.

Usage (when enabled):
    # In detection.py / FrameBus init:
    if is_broker_enabled():
        broker = InferenceBroker(model_path, device, conf, iou)
        broker.start()
        ...
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import queue
import time
from dataclasses import dataclass, field
from typing import Any, Optional

log = logging.getLogger(__name__)

_BROKER_ENABLED = os.getenv("INFERENCE_BROKER_ENABLED", "false").lower() in ("true", "1", "yes")
_BATCH_SIZE = max(1, int(os.getenv("INFERENCE_BATCH_SIZE", "4")))
_BATCH_TIMEOUT_MS = float(os.getenv("INFERENCE_BATCH_TIMEOUT_MS", "20"))


def is_broker_enabled() -> bool:
    """Return True when the inference broker is requested via env."""
    return _BROKER_ENABLED


@dataclass
class InferenceRequest:
    camera_id: str
    frame_id: int
    frame: Any          # np.ndarray BGR
    reply_queue: object  # multiprocessing.Queue for InferenceResult


@dataclass
class InferenceResult:
    camera_id: str
    frame_id: int
    results: Any        # Ultralytics Results list
    error: Optional[str] = None


def run_inference_broker(
    request_queue: multiprocessing.Queue,
    model_path: str,
    device: str,
    conf: float,
    iou: float,
    classes: Optional[list],
    max_det: int,
    tracker_yaml: str,
    stop_event: multiprocessing.Event,
) -> None:
    """
    Entry point for the broker subprocess.  Loads the YOLO model once and
    processes batched inference requests from *request_queue* until
    *stop_event* is set.
    """
    from ultralytics import YOLO

    _log = logging.getLogger("inference_broker")
    _log.info("[broker] Starting — model=%s device=%s batch=%d", model_path, device, _BATCH_SIZE)

    try:
        model = YOLO(model_path)
    except Exception as exc:
        _log.error("[broker] Failed to load model: %s", exc)
        return

    while not stop_event.is_set():
        batch: list[InferenceRequest] = []
        deadline = time.monotonic() + _BATCH_TIMEOUT_MS / 1000.0

        while len(batch) < _BATCH_SIZE and time.monotonic() < deadline:
            try:
                req = request_queue.get(timeout=max(0.001, deadline - time.monotonic()))
                batch.append(req)
            except queue.Empty:
                break

        if not batch:
            continue

        for req in batch:
            try:
                results = model.track(
                    req.frame,
                    persist=True,
                    tracker=tracker_yaml,
                    conf=conf,
                    iou=iou,
                    classes=classes,
                    device=device,
                    verbose=False,
                    max_det=max_det,
                )
                req.reply_queue.put(
                    InferenceResult(
                        camera_id=req.camera_id,
                        frame_id=req.frame_id,
                        results=results,
                    )
                )
            except Exception as exc:
                req.reply_queue.put(
                    InferenceResult(
                        camera_id=req.camera_id,
                        frame_id=req.frame_id,
                        results=[],
                        error=str(exc),
                    )
                )

    _log.info("[broker] Stopping.")


class InferenceBroker:
    """
    Lifecycle wrapper for the inference broker subprocess.
    Gated by INFERENCE_BROKER_ENABLED=true.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        conf: float = 0.25,
        iou: float = 0.45,
        classes: Optional[list] = None,
        max_det: int = 300,
        tracker_yaml: str = "botsort.yaml",
    ) -> None:
        self._model_path  = model_path
        self._device      = device
        self._conf        = conf
        self._iou         = iou
        self._classes     = classes
        self._max_det     = max_det
        self._tracker_yaml = tracker_yaml
        self._manager     = multiprocessing.Manager()
        self._request_q   = self._manager.Queue(maxsize=256)
        self._stop_event  = self._manager.Event()
        self._process: Optional[multiprocessing.Process] = None

    @property
    def request_queue(self) -> multiprocessing.Queue:
        return self._request_q

    def start(self) -> None:
        if self._process is not None and self._process.is_alive():
            return
        self._stop_event.clear()
        self._process = multiprocessing.Process(
            target=run_inference_broker,
            args=(
                self._request_q,
                self._model_path,
                self._device,
                self._conf,
                self._iou,
                self._classes,
                self._max_det,
                self._tracker_yaml,
                self._stop_event,
            ),
            daemon=True,
            name="inference_broker",
        )
        self._process.start()
        log.info("[InferenceBroker] Started (pid=%s)", self._process.pid)

    def stop(self) -> None:
        self._stop_event.set()
        if self._process is not None and self._process.is_alive():
            self._process.join(timeout=5.0)
            if self._process.is_alive():
                self._process.terminate()
        log.info("[InferenceBroker] Stopped.")
