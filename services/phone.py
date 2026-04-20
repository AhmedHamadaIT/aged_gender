"""
services/phone.py
-----------------
Phone detection service for person crops using a YOLO ONNX model.

Reads  context["data"]["frame"]
       context["data"]["detection"]["items"]  — List[Detection]
Writes context["data"]["use_case"]["phone"]   — List[PhoneResult]

If SAVE_OUTPUT=True, draws phone boxes + labels on the frame.
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Any

import cv2
import numpy as np
from dotenv import load_dotenv
from ultralytics import YOLO

from logger.logger_config import Logger

load_dotenv()

# Phone class mapping.
LABELS = {
    0: "phone",
}

# Person crop padding.
PADDING = int(os.getenv("PHONE_PADDING", "10"))

# Logger instance.
log = Logger.get_logger(__name__)

# Box colors for phone labels.
COLORS = {
    "phone": (0, 140, 255),
}

# ─────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────


@dataclass
class PhoneResult:
    person_bbox: tuple
    phone_detected: bool
    count: int
    items: List[Dict[str, Any]]

    def to_dict(self):
        return {
            "person_bbox": list(self.person_bbox),
            "phone_detected": self.phone_detected,
            "count": self.count,
            "items": self.items,
        }

# ─────────────────────────────────────────────
# Phone service
# ─────────────────────────────────────────────


class PhoneService:
    # Load ONNX model and runtime settings.
    def __init__(self):
        model_path = os.getenv("PHONE_MODEL", os.getenv(
            "PHONE_MODEL_PATH", "./models/best_phone.onnx"))
        self.conf = float(os.getenv("PHONE_CONF", "0.25"))
        self.iou = float(os.getenv("PHONE_IOU", "0.45"))
        self.save = os.getenv("SAVE_OUTPUT", "True").lower() in (
            "true", "1", "yes")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"[PHONE] Model not found: {model_path}")
        if not model_path.lower().endswith(".onnx"):
            raise ValueError("[PHONE] Model must be ONNX (.onnx)")

        log.info(f"[PHONE] Loading : {model_path}")
        log.info(f"[PHONE] Conf    : {self.conf}")
        log.info(f"[PHONE] IoU     : {self.iou}")

        self.model = YOLO(model_path, task="detect")

        log.info(f"[PHONE] Ready — labels: {list(LABELS.values())}\n")

    # Crop person box with small padding.
    def _crop_bbox(self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
        h, w = frame.shape[:2]
        x1 = max(0, x1 - PADDING)
        y1 = max(0, y1 - PADDING)
        x2 = min(w, x2 + PADDING)
        y2 = min(h, y2 + PADDING)
        return frame[y1:y2, x1:x2]

    # Run phone detection on one person crop.
    def _predict_crop(self, crop: np.ndarray, ox: int, oy: int) -> Dict[str, Any]:
        if crop is None or crop.size == 0:
            return {"count": 0, "items": []}

        try:
            results = self.model.predict(
                source=crop,
                conf=self.conf,
                iou=self.iou,
                verbose=False,
            )
        except Exception as exc:
            log.error(f"[PHONE] Inference failed: {exc}")
            return {"count": 0, "items": []}

        if not results:
            return {"count": 0, "items": []}

        items: List[Dict[str, Any]] = []
        result = results[0]
        boxes = getattr(result, "boxes", None)

        if boxes is None or len(boxes) == 0:
            return {"count": 0, "items": []}

        for box in boxes:
            try:
                class_id = int(box.cls.item())
                class_name = LABELS.get(class_id, f"class_{class_id}")
                conf = float(box.conf.item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # Convert crop-relative coords to full-frame coords.
                items.append(
                    {
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": round(conf, 4),
                        "x1": int(x1) + ox,
                        "y1": int(y1) + oy,
                        "x2": int(x2) + ox,
                        "y2": int(y2) + oy,
                    }
                )
            except Exception as exc:
                log.error(f"[PHONE] Failed to parse box: {exc}")

        return {"count": len(items), "items": items}

    # Process frame detections and attach phone results to context.
    def __call__(self, context: Dict[str, Any]) -> Dict[str, Any]:
        frame = context["data"]["frame"]
        detections = context["data"]["detection"].get("items", [])
        results: List[PhoneResult] = []

        for det in detections:
            crop = self._crop_bbox(frame, det.x1, det.y1, det.x2, det.y2)
            pred = self._predict_crop(crop, det.x1, det.y1)
            results.append(
                PhoneResult(
                    person_bbox=det.bbox,
                    phone_detected=pred["count"] > 0,
                    count=pred["count"],
                    items=pred["items"],
                )
            )

        context["data"]["use_case"]["phone"] = results

        # ── Draw phone annotations on frame if saving ──
        if self.save:
            context["data"]["frame"] = self._draw(frame, results)
        return context

    # Draw phone boxes and labels on frame.
    def _draw(self, frame: np.ndarray, results: List[PhoneResult]) -> np.ndarray:
        """
        draw phone predicted classes on each person detected
        """
        out = frame.copy()

        for r in results:
            for item in r.items:
                x1, y1 = item["x1"], item["y1"]
                x2, y2 = item["x2"], item["y2"]
                cls = item["class_name"]
                conf = item["confidence"]

                color = COLORS.get(cls, (255, 255, 255))
                label = f"{cls} {conf:.2f}"

                cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
                cv2.rectangle(out, (x1, y1 - th - 8),
                              (x1 + tw + 4, y1), color, -1)
                cv2.putText(
                    out, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                    (255, 255, 255), 1, cv2.LINE_AA,
                )

        return out
