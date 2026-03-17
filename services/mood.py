"""
services/mood.py
----------------
Mood / Emotion service using best_mood.onnx (ONNX export of YOLOv8 3-class model).

Classes:  0=Angry  1=Happy  2=Neutral 

Reads  context["data"]["frame"]
       context["data"]["detection"]["items"]
Writes context["data"]["use_case"]["mood"]  — List[MoodResult]
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Any

import cv2
import numpy as np


# The labels based on the user's dataset mapping
MOOD_LABELS = ["Angry", "Happy", "Neutral"]

MOOD_COLORS = {
    "Angry"   : (50, 50, 255),    # Red in BGR
    "Happy"   : (0, 215, 255),    # Yellow/Gold in BGR
    "Neutral" : (200, 200, 200),  # Gray
}

@dataclass
class MoodResult:
    bbox      : tuple
    mood      : str
    confidence: float

    def to_dict(self):
        return {
            "bbox"      : list(self.bbox),
            "mood"      : self.mood,
            "confidence": round(self.confidence, 4),
        }


class MoodService:
    def __init__(self):
        model_path = os.getenv("MOOD_MODEL", "./models/best_mood.onnx")
        self.save  = os.getenv("SAVE_OUTPUT", "True").lower() in ("true", "1", "yes")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"[MOOD] Model not found: {model_path}")

        import onnxruntime as ort
        self.sess     = ort.InferenceSession(model_path)
        self.inp_name = self.sess.get_inputs()[0].name
        inp_shape     = self.sess.get_inputs()[0].shape   # e.g. [1,3,128,128]
        self.img_size = (int(inp_shape[3]), int(inp_shape[2]))   # (W, H)

        print(f"[MOOD] Ready — input {self.img_size}, classes: {MOOD_LABELS}\n")

    # ── Preprocessing ──────────────────────────────────────────────────────
    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        img = cv2.resize(crop, self.img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[np.newaxis, ...]   # HWC→NCHW
        return img

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / e.sum()

    # ── Inference ──────────────────────────────────────────────────────────
    def _predict(self, crop: np.ndarray):
        if crop.size == 0:
            return "Unknown", 0.0
            
        # Crop to the top half as the model is trained mainly from chest to face
        h = crop.shape[0]
        half_crop = crop[:h // 2, :]
        if half_crop.size == 0:
            half_crop = crop
            
        blob    = self._preprocess(half_crop)
        outputs = self.sess.run(None, {self.inp_name: blob})
        probs   = self._softmax(outputs[0][0])   # shape (num_classes,)
        idx     = int(np.argmax(probs))
        return MOOD_LABELS[idx], float(probs[idx])

    # ── Service callable ───────────────────────────────────────────────────
    def __call__(self, context: Dict[str, Any]) -> Dict[str, Any]:
        frame      = context["data"]["frame"]
        detections = context["data"]["detection"].get("items", [])
        results    = []

        for det in detections:
            crop        = frame[det.y1:det.y2, det.x1:det.x2]
            mood, conf  = self._predict(crop)
            results.append(MoodResult(bbox=det.bbox, mood=mood, confidence=conf))

        context["data"]["use_case"]["mood"] = results

        if self.save:
            context["data"]["frame"] = self._draw(frame, results)

        return context

    # ── Annotation ─────────────────────────────────────────────────────────
    def _draw(self, frame: np.ndarray, results: List[MoodResult]) -> np.ndarray:
        out = frame.copy()
        for r in results:
            x1, y1, x2, y2 = r.bbox
            color = MOOD_COLORS.get(r.mood, (255, 255, 255))
            label = f"{r.mood} {r.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            # draw label strip above the bbox top line
            cv2.rectangle(out, (x1, y1 - th - 24), (x1 + tw + 4, y1 - 8), color, -1)
            cv2.putText(out, label, (x1 + 2, y1 - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        return out
