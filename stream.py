"""
stream.py
---------
Frame sourcing — RTSP stream or local video.
Exposes a single generator: frames(source)

frames(rtsp_url)   → reads directly from RTSP
frames(video_path) → reads from local video file
frames()           → uses USE_STREAM / RTSP_URL_1 from .env
"""

import os
import time

import cv2
from dotenv import load_dotenv

load_dotenv()

USE_STREAM  = os.getenv("USE_STREAM",   "True").lower() in ("true", "1", "yes")
RTSP_URL    = os.getenv("CAMERA_1_URL", "")
INPUT_VIDEO = os.getenv("INPUT_VIDEO",  "./videos/sample.mp4")


class _RTSPReader:
    def __init__(self, url: str):
        self.url = url
        self.cap = None

    def connect(self):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        print(f"[STREAM] Connecting: {self.url}")
        self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError(f"[STREAM] Cannot open: {self.url}")
        w   = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"[STREAM] Connected — {w}x{h} @ {fps:.1f}fps")

    def read_frame(self):
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        if self.cap:
            self.cap.release()


class _VideoReader:
    def __init__(self, path: str):
        self.path = path
        self.cap  = None

    def connect(self):
        print(f"[VIDEO] Opening: {self.path}")
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise RuntimeError(f"[VIDEO] Cannot open: {self.path}")

    def read_frame(self):
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        if self.cap:
            self.cap.release()


def frames(source: str = None):
    """
    Generator yielding BGR numpy frames.

    Args:
        source: RTSP URL, video file path, or None (uses .env defaults)
    """
    if source is None:
        source = RTSP_URL if USE_STREAM else INPUT_VIDEO

    is_rtsp = source.startswith("rtsp://")
    reader  = _RTSPReader(source) if is_rtsp else _VideoReader(source)

    consecutive_fails = 0
    max_fails         = 10

    reader.connect()
    try:
        while True:
            frame = reader.read_frame()
            if frame is None:
                consecutive_fails += 1
                if consecutive_fails >= max_fails:
                    if is_rtsp:
                        print(f"[STREAM] Reconnecting...")
                        reader.release()
                        time.sleep(2)
                        reader.connect()
                        consecutive_fails = 0
                    else:
                        break  # end of video file
                continue
            consecutive_fails = 0
            yield frame
    finally:
        reader.release()