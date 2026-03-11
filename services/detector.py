"""
services/detector.py
--------------------
YOLO detection service.

Reads  context["data"]["frame"]
Writes context["data"]["detection"]

If SAVE_OUTPUT=True, draws bounding boxes onto context["data"]["frame"]
so downstream services and the final save see the annotated frame.

Output:
{
    "detection": {
        "items": List[Detection],
        "count": int
    }
}
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Any

import cv2
import numpy as np
from dotenv import load_dotenv
from ultralytics import YOLO

from logger.logger_config import Logger
import os
from dotenv import load_dotenv
load_dotenv()
log = Logger.get_logger(__name__)

COLORS = [
    ( 56, 193, 114), ( 52, 152, 219), (231,  76,  60),
    (241, 196,  15), (155,  89, 182), ( 26, 188, 156),
    (230, 126,  34), ( 52,  73,  94),
]


# ─────────────────────────────────────────────
# Detection dataclass
# ─────────────────────────────────────────────
@dataclass
class Detection:
    x1        : int
    y1        : int
    x2        : int
    y2        : int
    class_id  : int
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
            "bbox"      : list(self.bbox),
            "class_id"  : self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "center"    : list(self.center),
            "width"     : self.width,
            "height"    : self.height,
        }


# ─────────────────────────────────────────────
# Detector service
# ─────────────────────────────────────────────
class DetectorService:
    def __init__(self):
        model_path   = os.getenv("YOLO_MODEL",        "yolov8n.pt")
        self.conf    = float(os.getenv("CONF_THRESHOLD", "0.35"))
        _device_raw  = os.getenv("DEVICE", "0")
        self.device  = int(_device_raw) if _device_raw.isdigit() else _device_raw
        _classes_raw = os.getenv("FILTER_CLASSES", "")
        self.classes = [int(c.strip()) for c in _classes_raw.split(",") if c.strip()] or None
        self.save    = os.getenv("SAVE_OUTPUT", "True").lower() in ("true", "1", "yes")

        log.info(f"[DETECTOR] Loading : {model_path}")
        log.info(f"[DETECTOR] Device  : {self.device}")
        log.info(f"[DETECTOR] Conf    : {self.conf}")
        log.info(f"[DETECTOR] Classes : {self.classes or 'All'}")

        self.model = YOLO(model_path, task="detect")
        self.names = self.model.names

        log.info(f"[DETECTOR] Ready — {len(self.names)} classes\n")

    def __call__(self, context: Dict[str, Any]) -> Dict[str, Any]:
        frame   = context["data"]["frame"]
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
                cls_id         = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append(Detection(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    class_id=cls_id,
                    class_name=self.names[cls_id],
                    confidence=score,
                ))

        context["data"]["detection"] = {
            "items": detections,
            "count": len(detections),
        }

        # ── Draw boxes onto frame if saving ──
        # Downstream services (e.g. AgeGenderService) will draw on top of this
        if self.save:
            context["data"]["frame"] = self._draw(frame, detections)

        return context

    def _draw(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw bounding boxes and class labels onto the frame."""
        out = frame.copy()
        for det in detections:
            color = COLORS[det.class_id % len(COLORS)]
            label = f"{det.class_name} {det.confidence:.2f}"

            cv2.rectangle(out, (det.x1, det.y1), (det.x2, det.y2), color, 2)

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(out, (det.x1, det.y1 - th - 8), (det.x1 + tw + 4, det.y1), color, -1)
            cv2.putText(
                out, label, (det.x1 + 2, det.y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA,
            )
        return out