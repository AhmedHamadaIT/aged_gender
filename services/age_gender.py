"""
services/age_gender.py
----------------------
Age & Gender classification service using a dual-head MobileNetV3 ONNX model.

Model:      best_aged_gender_6.onnx
Input:      (1, 3, 224, 224) — normalized RGB
Outputs:
    gender_logits  (1, 2)  → [Female, Male]
    age_output     (1, 4)  → [Young, MiddleAged, Senior, Elderly]

Reads  context["data"]["frame"]
       context["data"]["detection"]["items"]  — List[Detection]

Writes context["data"]["use_case"]["age_gender"] — List[AgeGenderResult]

If SAVE_OUTPUT=True, draws gender + age label below each bbox on the frame.
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Any

import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ── Label maps ────────────────────────────────
GENDER_LABELS = ["Female", "Male"]
AGE_LABELS    = ["Young", "MiddleAged", "Senior", "Elderly"]

PADDING = int(os.getenv("AGE_GENDER_PADDING", "10"))


# ─────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────
@dataclass
class AgeGenderResult:
    bbox      : tuple
    gender    : str    # "Male" | "Female"
    age_group : str    # "Young" | "MiddleAged" | "Senior" | "Elderly"
    confidence: float  # confidence of the top gender prediction

    def to_dict(self):
        return {
            "bbox"      : list(self.bbox),
            "gender"    : self.gender,
            "age_group" : self.age_group,
            "confidence": round(self.confidence, 4),
        }


# ─────────────────────────────────────────────
# Age & Gender service
# ─────────────────────────────────────────────
class AgeGenderService:
    def __init__(self):
        model_path = os.getenv("AGE_GENDER_MODEL", "./models/best_aged_gender_6.onnx")
        self.save  = os.getenv("SAVE_OUTPUT", "True").lower() in ("true", "1", "yes")

        print(f"[AGE_GENDER] Loading : {model_path}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"[AGE_GENDER] Model not found: {model_path}")

        import onnxruntime as ort
        self.sess      = ort.InferenceSession(model_path)
        self.inp_name  = self.sess.get_inputs()[0].name   # "input"

        print(f"[AGE_GENDER] Ready — gender: {GENDER_LABELS}, age: {AGE_LABELS}\n")

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        """Resize, normalize, and format crop for model input."""
        img = cv2.resize(crop, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img  = (img - mean) / std
        # HWC → NCHW
        img  = np.transpose(img, (2, 0, 1))[np.newaxis, ...]
        return img

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / e.sum()

    def _crop_bbox(self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
        """Crop bbox with padding."""
        h, w = frame.shape[:2]
        x1 = max(0, x1 - PADDING)
        y1 = max(0, y1 - PADDING)
        x2 = min(w, x2 + PADDING)
        y2 = min(h, y2 + PADDING)
        return frame[y1:y2, x1:x2]

    def _predict(self, crop: np.ndarray):
        """
        Run inference on a single bbox crop.
        Returns (gender, age_group, confidence).
        """
        if crop.size == 0:
            return "Unknown", "Unknown", 0.0

        blob    = self._preprocess(crop)
        outputs = self.sess.run(None, {self.inp_name: blob})

        gender_probs = self._softmax(outputs[0][0])  # (2,)
        age_probs    = self._softmax(outputs[1][0])  # (4,)

        gender_idx   = int(np.argmax(gender_probs))
        age_idx      = int(np.argmax(age_probs))

        return (
            GENDER_LABELS[gender_idx],
            AGE_LABELS[age_idx],
            float(gender_probs[gender_idx]),
        )

    def __call__(self, context: Dict[str, Any]) -> Dict[str, Any]:
        frame      = context["data"]["frame"]
        detections = context["data"]["detection"].get("items", [])
        results    = []

        for det in detections:
            crop              = self._crop_bbox(frame, det.x1, det.y1, det.x2, det.y2)
            gender, age, conf = self._predict(crop)
            results.append(AgeGenderResult(
                bbox       = det.bbox,
                gender     = gender,
                age_group  = age,
                confidence = conf,
            ))

        context["data"]["use_case"]["age_gender"] = results

        # ── Draw labels on frame if saving ──
        if self.save:
            context["data"]["frame"] = self._draw(frame, results)

        return context

    def _draw(self, frame: np.ndarray, results: List[AgeGenderResult]) -> np.ndarray:
        """Draw gender + age_group label below each detection bbox."""
        out = frame.copy()
        for r in results:
            x1, y1, x2, y2 = r.bbox
            label = f"{r.gender} | {r.age_group}"
            color = (255, 165, 0) if r.gender == "Male" else (255, 105, 180)

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(out, (x1, y2), (x1 + tw + 4, y2 + th + 8), color, -1)
            cv2.putText(
                out, label, (x1 + 2, y2 + th + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA,
            )
        return out