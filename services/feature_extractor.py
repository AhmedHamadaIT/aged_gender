"""
services/feature_extractor.py
-----------------------------
Person Re-Identification service using OSNet feature extraction + Qdrant identity gallery.

Reads  context["data"]["frame"]
       context["data"]["detection"]["items"]  — List[Detection]

Writes context["data"]["use_case"]["reid"]    — List[ReIDResult]

If SAVE_OUTPUT=True, draws person ID labels below each bbox on the frame.

Environment variables:
    REID_MODEL_PATH       — Path to JIT-traced OSNet model (default: models/osnet_x1_0.pt)
    DEVICE                — Inference device                (default: cpu)
    REID_AUTO_REGISTER    — Auto-enroll unknown persons     (default: True)
    REID_PADDING          — Bbox padding in pixels          (default: 10)
    SAVE_OUTPUT           — Draw annotations on frame       (default: True)
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from dotenv import load_dotenv

from logger.logger_config import Logger
from store.identity_manager import IdentityManager

load_dotenv()
log = Logger.get_logger(__name__)

PADDING = int(os.getenv("REID_PADDING", "10"))


# ─────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────
@dataclass
class ReIDResult:
    bbox      : tuple
    person_id : str
    confidence: float
    is_new    : bool     # True if this person was just enrolled

    def to_dict(self):
        return {
            "bbox"      : list(self.bbox),
            "person_id" : self.person_id,
            "confidence": round(self.confidence, 4),
            "is_new"    : self.is_new,
        }


# ─────────────────────────────────────────────
# ReID service
# ─────────────────────────────────────────────
class ReIDService:
    def __init__(self):
        model_path        = os.getenv("REID_MODEL_PATH", "models/osnet_x1_0.pt")
        _device_raw       = os.getenv("DEVICE", "cpu")
        self.device       = int(_device_raw) if _device_raw.isdigit() else _device_raw
        self.auto_register = os.getenv("REID_AUTO_REGISTER", "True").lower() in ("true", "1", "yes")
        self.save          = os.getenv("SAVE_OUTPUT", "True").lower() in ("true", "1", "yes")

        log.info(f"[REID] Loading model : {model_path}")
        log.info(f"[REID] Device        : {self.device}")
        log.info(f"[REID] Auto-register : {self.auto_register}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"[REID] Model not found: {model_path}")

        self.model = torch.jit.load(model_path, map_location=str(self.device))
        self.model.eval()

        self.transform = T.Compose([
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Initialize identity store
        self.identity_manager = IdentityManager()

        log.info("[REID] Ready\n")

    # ── Pipeline entry point ─────────────────

    def __call__(self, context: Dict[str, Any]) -> Dict[str, Any]:
        frame      = context["data"]["frame"]
        detections = context["data"]["detection"].get("items", [])
        results    = []

        for det in detections:
            crop = self._crop_bbox(frame, det.x1, det.y1, det.x2, det.y2)
            if crop.size == 0:
                continue

            # Extract feature vector
            feature = self._extract(crop)
            if feature is None:
                continue

            # Identify against gallery
            match = self.identity_manager.identify(feature)

            if match is not None:
                results.append(ReIDResult(
                    bbox       = det.bbox,
                    person_id  = match["person_id"],
                    confidence = match["confidence"],
                    is_new     = False,
                ))
            elif self.auto_register:
                new_id = self.identity_manager.register(feature)
                results.append(ReIDResult(
                    bbox       = det.bbox,
                    person_id  = new_id,
                    confidence = 1.0,
                    is_new     = True,
                ))
            # else: skip — unknown person, auto-register disabled

        context["data"]["use_case"]["reid"] = results

        # ── Draw labels on frame if saving ──
        if self.save:
            context["data"]["frame"] = self._draw(frame, results)

        return context

    # ── Internal methods ─────────────────────

    def _crop_bbox(self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
        """Crop bbox region with padding, clamped to frame bounds."""
        h, w = frame.shape[:2]
        x1 = max(0, x1 - PADDING)
        y1 = max(0, y1 - PADDING)
        x2 = min(w, x2 + PADDING)
        y2 = min(h, y2 + PADDING)
        return frame[y1:y2, x1:x2]

    def _extract(self, cropped_bgr: np.ndarray) -> Optional[list]:
        """Extract a normalized feature vector from a BGR crop. Returns None on error."""
        try:
            img_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
            tensor  = self.transform(Image.fromarray(img_rgb)).unsqueeze(0).to(self.device)

            with torch.no_grad():
                feature = self.model(tensor)[0]

            feat_norm = F.normalize(feature, p=2, dim=0)
            return feat_norm.cpu().numpy().tolist()
        except Exception as e:
            log.warning(f"[REID] Feature extraction failed: {e}")
            return None

    def _draw(self, frame: np.ndarray, results: List[ReIDResult]) -> np.ndarray:
        """Draw person ID labels below each detection bbox."""
        out = frame.copy()
        for r in results:
            x1, y1, x2, y2 = r.bbox
            # Use short ID for readability (first 8 chars of UUID)
            short_id = r.person_id[:8] if len(r.person_id) > 8 else r.person_id
            label    = f"ID:{short_id}"
            color    = (0, 255, 0) if not r.is_new else (0, 165, 255)  # green=known, orange=new

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(out, (x1, y2), (x1 + tw + 4, y2 + th + 8), color, -1)
            cv2.putText(
                out, label, (x1 + 2, y2 + th + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA,
            )
        return out