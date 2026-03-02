"""
apis/detection.py
-----------------
Detection resource — business logic only, no routing.
Routes are registered in app.py.
"""

import os
import time
import threading
from typing import Optional

from fastapi import HTTPException
from dotenv import load_dotenv

from apis.base import BaseResource
from services.detector import DetectorService
from stream import frames, USE_STREAM, INPUT_VIDEO, RTSP_URL
from utils  import annotate, resize, save_frame
from schemas import DetectionRequest, DetectionStatus

load_dotenv()

# ─────────────────────────────────────────────
# State
# ─────────────────────────────────────────────
_state = DetectionStatus(
    running=False, frame_count=0, fps=0.0,
    last_detections=0, total_detections=0,
    uptime_seconds=None, error=None,
)
_thread: Optional[threading.Thread] = None
_stop_event = threading.Event()


# ─────────────────────────────────────────────
# Resource
# ─────────────────────────────────────────────
class DetectionResource(BaseResource):
    def __init__(self):
        super().__init__()
        self.class_instance = {
            "start": self._start,
            "stop" : self._stop,
        }
        self._detector: Optional[DetectorService] = None
        self._save    = os.getenv("SAVE_OUTPUT", "True").lower() in ("true", "1", "yes")
        self._out_dir = os.getenv("OUTPUT_DIR",  "./outputs/frames")
        self._width   = int(os.getenv("WIDTH"))
        self._height  = int(os.getenv("HEIGHT"))

    def init(self):
        self._detector = DetectorService()

    def on_post(self, req: DetectionRequest):
        return self.get_service(req.action)()

    def on_get(self):
        return _state

    # ── Actions ──────────────────────────────

    def _start(self):
        global _thread
        if _state.running:
            raise HTTPException(status_code=409, detail="Detection already running.")

        _stop_event.clear()
        _thread = threading.Thread(target=self._worker, daemon=True)
        _thread.start()

        return {
            "status": "started",
            "source": RTSP_URL if USE_STREAM else INPUT_VIDEO,
            "save"  : self._save,
            "output": self._out_dir if self._save else None,
        }

    def _stop(self):
        if not _state.running:
            raise HTTPException(status_code=409, detail="Detection is not running.")

        _stop_event.set()
        _thread.join(timeout=5)

        return {
            "status"           : "stopped",
            "frames_processed" : _state.frame_count,
            "total_detections" : _state.total_detections,
        }

    # ── Worker ───────────────────────────────

    def _worker(self):
        global _state
        _state = DetectionStatus(
            running=True, frame_count=0, fps=0.0,
            last_detections=0, total_detections=0,
            uptime_seconds=None, error=None,
        )

        fps_counter = 0
        fps_timer   = time.time()
        started_at  = time.time()
        mode_label  = "RTSP Stream" if USE_STREAM else "Local Video"

        try:
            if self._save:
                os.makedirs(self._out_dir, exist_ok=True)

            for frame in frames():
                if _stop_event.is_set():
                    break

                _state.frame_count += 1
                fps_counter        += 1

                elapsed = time.time() - fps_timer
                if elapsed >= 1.0:
                    _state.fps   = round(fps_counter / elapsed, 2)
                    fps_counter  = 0
                    fps_timer    = time.time()

                _state.uptime_seconds = round(time.time() - started_at, 1)

                infer_frame              = resize(frame, self._width, self._height)
                detections               = self._detector(infer_frame)
                _state.last_detections   = len(detections)
                _state.total_detections += len(detections)

                if self._save:
                    annotated = annotate(infer_frame, detections, _state.fps, mode_label)
                    save_frame(annotated, self._out_dir, _state.frame_count)

        except Exception as e:
            _state.error = str(e)
            print(f"[DETECTION] Worker error: {e}")
        finally:
            _state.running = False
            print(f"[DETECTION] Stopped. Frames: {_state.frame_count}")


# ── Singleton ─────────────────────────────────
detection = DetectionResource()