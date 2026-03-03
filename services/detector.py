"""
services/detector.py
--------------------
YOLO detection service.

Reads frame from context, adds detections to context.

Input context:  {"frame": np.ndarray, ...}
Output context: {"frame": np.ndarray, "detections": List[Detection], ...}
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()


# ─────────────────────────────────────────────
# Detection result
# ─────────────────────────────────────────────
@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    class_id: int
    class_name: str
    confidence: float

    @property
    def bbox(self):
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1

    @property
    def center(self):
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    def to_dict(self):
        return {
            "bbox"       : list(self.bbox),
            "class_id"   : self.class_id,
            "class_name" : self.class_name,
            "confidence" : round(self.confidence, 4),
            "center"     : list(self.center),
            "width"      : self.width,
            "height"     : self.height,
        }


# ─────────────────────────────────────────────
# Detector service
# ─────────────────────────────────────────────
class DetectorService:
    """
    YOLO object detection service.

    Reads "frame" from context dict, appends "detections".

    Usage in pipeline:
        context = detector(context)
        detections = context["detections"]
    """

    def __init__(self):
        model_path   = os.getenv("YOLO_MODEL",       "yolov8n.pt")
        self.conf    = float(os.getenv("CONF_THRESHOLD", "0.35"))
        _device_raw  = os.getenv("DEVICE", "0")
        self.device  = int(_device_raw) if _device_raw.isdigit() else _device_raw
        _classes_raw = os.getenv("FILTER_CLASSES", "")
        self.classes = [int(c.strip()) for c in _classes_raw.split(",") if c.strip()] or None

        print(f"[DETECTOR] Loading : {model_path}")
        print(f"[DETECTOR] Device  : {self.device}")
        print(f"[DETECTOR] Conf    : {self.conf}")
        print(f"[DETECTOR] Classes : {self.classes or 'All'}")

        self.model = YOLO(model_path)
        self.names = self.model.names

        print(f"[DETECTOR] Ready — {len(self.names)} classes\n")

    def __call__(self, context: Dict[str, Any]) -> Dict[str, Any]:
        frame   = context["frame"]
        results = self.model.predict(
            frame,
            conf    = self.conf,
            device  = self.device,
            classes = self.classes,
            verbose = False,
        )

        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                score = float(box.conf[0])
                if score < self.conf:
                    continue
                cls_id      = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append(Detection(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    class_id=cls_id,
                    class_name=self.names[cls_id],
                    confidence=score,
                ))

        return {**context, "detections": detections}


YOLODetector = DetectorService