"""
stream.py
---------
Frame sourcing — RTSP stream or local video.
Exposes a single generator: frames()

USE_STREAM=True  → reads directly from RTSP via OpenCV
USE_STREAM=False → reads from local video file
"""

import os
import time

import cv2
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
USE_STREAM  = os.getenv("USE_STREAM",  "False").lower() in ("true", "1", "yes")
RTSP_URL    = os.getenv("RTSP_URL_1",  "rtsp://admin:Aa112233@10.0.3.71:554/Streaming/channels/1101")
INPUT_VIDEO = os.getenv("INPUT_VIDEO", "./videos/sample.mp4")
MJPEG_URL   = os.getenv("MJPEG_URL",   "")   # kept for backward compat, unused


# ─────────────────────────────────────────────
# RTSP Reader
# ─────────────────────────────────────────────
class _RTSPReader:
    """Reads frames directly from RTSP using OpenCV."""

    def __init__(self, url: str):
        self.url = url
        self.cap = None

    def connect(self):
        print(f"[STREAM] Connecting to RTSP: {self.url}")
        # Use TCP transport for stability (more reliable than UDP on networks)
        self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # minimize buffer lag
        if not self.cap.isOpened():
            raise RuntimeError(f"[STREAM] Cannot open RTSP: {self.url}")
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        w   = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[STREAM] Connected — {w}x{h} @ {fps:.1f}fps")

    def read_frame(self):
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        if self.cap:
            self.cap.release()
            print("[STREAM] Disconnected")


# ─────────────────────────────────────────────
# Local Video Reader
# ─────────────────────────────────────────────
class _VideoReader:
    """Reads frames from a local video file using OpenCV."""

    def __init__(self, path: str):
        self.path = path
        self.cap  = None

    def connect(self):
        print(f"[VIDEO] Opening: {self.path}")
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise RuntimeError(f"[VIDEO] Cannot open file: {self.path}")
        total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps   = self.cap.get(cv2.CAP_PROP_FPS)
        w     = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h     = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[VIDEO] Ready — {w}x{h} @ {fps:.1f}fps | {total} frames")

    def read_frame(self):
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        if self.cap:
            self.cap.release()
            print("[VIDEO] Released")


# ─────────────────────────────────────────────
# Public generator
# ─────────────────────────────────────────────
def frames():
    """
    Generator yielding BGR numpy frames from the configured source.

    USE_STREAM=True  → direct RTSP (full camera FPS, no middleware)
    USE_STREAM=False → local video file (stops at EOF)
    """
    if USE_STREAM:
        reader             = _RTSPReader(RTSP_URL)
        consecutive_fails  = 0
        max_fails          = 10

        reader.connect()
        try:
            while True:
                frame = reader.read_frame()
                if frame is None:
                    consecutive_fails += 1
                    print(f"[STREAM] Empty frame ({consecutive_fails}/{max_fails})")
                    if consecutive_fails >= max_fails:
                        print("[STREAM] Too many failures, reconnecting...")
                        reader.release()
                        time.sleep(2)
                        reader.connect()
                        consecutive_fails = 0
                    continue
                consecutive_fails = 0
                yield frame
        finally:
            reader.release()

    else:
        reader = _VideoReader(INPUT_VIDEO)
        reader.connect()
        try:
            while True:
                frame = reader.read_frame()
                if frame is None:
                    break
                yield frame
        finally:
            reader.release()