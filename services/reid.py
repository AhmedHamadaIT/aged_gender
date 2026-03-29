"""
services/feature_extractor.py
-----------------------------
Event-Based Person Re-Identification using OSNet + Qdrant.
Triggers identification only when a local track ends.
"""

import os
import gdown
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

PADDING             = int(os.getenv("REID_PADDING", "10"))
MAX_TRACK_AGE       = int(os.getenv("MAX_TRACK_AGE", "30"))      # Frames to wait before confirming track is gone
MAX_CROPS_PER_TRACK = int(os.getenv("MAX_CROPS_PER_TRACK", "5")) # Buffer size per track


@dataclass
class ReIDEvent:
    track_id  : int
    person_id : str
    confidence: float
    is_new    : bool
    last_bbox : tuple

    def to_dict(self):
        return {
            "track_id"  : self.track_id,
            "person_id" : self.person_id,
            "confidence": round(self.confidence, 4),
            "is_new"    : self.is_new,
            "last_bbox" : list(self.last_bbox)
        }


class ReIDService:
    def __init__(self):
        model_path = os.getenv("REID_MODEL_PATH", "models/osnet_x1_0.pt")
        # ... (Keep your existing download and model loading logic here) ...
        
        self.device = torch.device(os.getenv("DEVICE", "cpu"))
        self.model = torch.jit.load(model_path, map_location=str(self.device))
        self.model.eval()

        self.transform = T.Compose([
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.identity_manager = IdentityManager()
        self.auto_register    = os.getenv("REID_AUTO_REGISTER", "True").lower() in ("true", "1", "yes")

        # ── State Management for Tracking ──
        self.active_tracks = {}  # { track_id: {"crops": [(area, img)], "last_seen": int, "bbox": tuple} }
        self.frame_count   = 0

        log.info("[REID] Tracking Buffer Ready\n")


    def __call__(self, context: Dict[str, Any]) -> Dict[str, Any]:
        self.frame_count += 1
        frame      = context["data"]["frame"]
        detections = context["data"]["detection"].get("items", [])
        
        events = []

        # ─────────────────────────────────────────────
        # 1. Update Track Buffer
        # ─────────────────────────────────────────────
        for det in detections:
            track_id = getattr(det, "track_id", None)
            if track_id is None:
                continue

            crop = self._crop_bbox(frame, det.x1, det.y1, det.x2, det.y2)
            if crop.size == 0:
                continue

            # We use bounding box area as a simple proxy for "image quality/closeness"
            area = (det.x2 - det.x1) * (det.y2 - det.y1)

            if track_id not in self.active_tracks:
                self.active_tracks[track_id] = {
                    "crops": [], 
                    "last_seen": self.frame_count,
                    "bbox": det.bbox
                }

            track_data = self.active_tracks[track_id]
            track_data["last_seen"] = self.frame_count
            track_data["bbox"]      = det.bbox
            
            # Store crop, sort by area (largest first), keep top N
            track_data["crops"].append((area, crop))
            track_data["crops"].sort(key=lambda x: x[0], reverse=True)
            track_data["crops"] = track_data["crops"][:MAX_CROPS_PER_TRACK]


        # ─────────────────────────────────────────────
        # 2. Check for Lost Tracks & Trigger ReID
        # ─────────────────────────────────────────────
        lost_track_ids = []
        for tid, data in self.active_tracks.items():
            if self.frame_count - data["last_seen"] > MAX_TRACK_AGE:
                lost_track_ids.append(tid)

        for tid in lost_track_ids:
            track_data = self.active_tracks.pop(tid)
            
            # Extract features for all saved crops
            features = []
            for _, crop in track_data["crops"]:
                tensor_feat = self._extract_raw_tensor(crop)
                if tensor_feat is not None:
                    features.append(tensor_feat)
            
            if not features:
                continue

            # Stack, Average, and Re-Normalize
            stacked_features = torch.stack(features)           # Shape: [N, Embedding_Dim]
            avg_feature      = torch.mean(stacked_features, dim=0) # Shape: [Embedding_Dim]
            final_embedding  = F.normalize(avg_feature, p=2, dim=0).cpu().numpy().tolist()

            # Identify against Qdrant gallery
            match = self.identity_manager.identify(final_embedding)

            if match is not None:
                events.append(ReIDEvent(
                    track_id   = tid,
                    person_id  = match["person_id"],
                    confidence = match["confidence"],
                    is_new     = False,
                    last_bbox  = track_data["bbox"]
                ))
            elif self.auto_register:
                new_id = self.identity_manager.register(final_embedding)
                events.append(ReIDEvent(
                    track_id   = tid,
                    person_id  = new_id,
                    confidence = 1.0,
                    is_new     = True,
                    last_bbox  = track_data["bbox"]
                ))

        # This will now contain a list of completed visit events, not frame-by-frame detections
        context["data"]["use_case"]["reid_events"] = events
        return context

    # ── Internal methods ─────────────────────

    def _crop_bbox(self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1 - PADDING), max(0, y1 - PADDING)
        x2, y2 = min(w, x2 + PADDING), min(h, y2 + PADDING)
        return frame[y1:y2, x1:x2]

    def _extract_raw_tensor(self, cropped_bgr: np.ndarray) -> Optional[torch.Tensor]:
        """Returns the un-normalized tensor directly for mathematical averaging."""
        try:
            img_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
            tensor  = self.transform(Image.fromarray(img_rgb)).unsqueeze(0).to(self.device)

            with torch.no_grad():
                feature = self.model(tensor)[0]
            return feature
        except Exception as e:
            log.warning(f"[REID] Feature extraction failed: {e}")
            return None