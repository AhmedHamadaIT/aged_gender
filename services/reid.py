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
    track_id: int
    image_path: str
    frame_number: int   


class ReIDService:
    def __init__(self):
        model_path = os.getenv("REID_MODEL_PATH", "models/osnet_x1_0.pt")        
        self.device = torch.device(os.getenv("DEVICE", "cpu"))
        self.model = torch.jit.load(model_path, map_location=str(self.device))
        self.model.eval()

        self.transform = T.Compose([
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.identity_manager = IdentityManager()
        
        self.gallery_dir = os.getenv("GALLERY_DIR", "./gallery")
        os.makedirs(self.gallery_dir, exist_ok=True)

        # 30 frames = 1 second (assuming 30fps video)
        self.throttle_frames = int(os.getenv("REID_THROTTLE", "30")) 
        self.padding = int(os.getenv("REID_PADDING", "10"))
                           
        self.last_extracted = {}  # { track_id: frame_number }
        self.frame_count = 0

        log.info("[REID] Tracking Buffer Ready\n")


    def __call__(self, context: Dict[str, Any]) -> Dict[str, Any]:
        self.frame_count += 1
        frame      = context["data"]["frame"]
        detections = context["data"]["detection"].get("items", [])

        clean_frame = context["data"].get("clean_frame", frame)

        
        events = []
        current_track_ids = set()

        # ─────────────────────────────────────────────
        # 1. Update Track Buffer
        # ─────────────────────────────────────────────
        for det in detections:
            track_id = getattr(det, "track_id", None)
            if track_id is None:
                continue

            current_track_ids.add(track_id)

            # If this is a new person, set their last extracted frame far in the past 
            # so they trigger an extraction immediately on frame 1.
            if track_id not in self.last_extracted:
                self.last_extracted[track_id] = -self.throttle_frames

            frames_since_last = self.frame_count - self.last_extracted[track_id]

            if frames_since_last >= self.throttle_frames:
                
                # 1. Crop
                crop = self._crop_bbox(clean_frame, det.x1, det.y1, det.x2, det.y2)
                if crop.size == 0:
                    continue
                
                # 2. Extract Embedding
                tensor_feat = self._extract_raw_tensor(crop)
                if tensor_feat is None:
                    continue
                
                final_embedding = F.normalize(tensor_feat, p=2, dim=0).cpu().numpy().tolist()

                # 3. Save Image locally
                img_filename = f"track_{track_id}_frame_{self.frame_count}.jpg"
                img_path = os.path.join(self.gallery_dir, img_filename)
                cv2.imwrite(img_path, crop)

                # 4. Insert to Vector DB with Metadata
                metadata = {
                    "image_path": img_path, 
                    "track_id": track_id,
                    "frame": self.frame_count
                }
                # NOTE: You will need to ensure your IdentityManager handles passing this metadata dict to Qdrant!
                self.identity_manager.register(final_embedding, payload=metadata)

                # 5. Update State
                self.last_extracted[track_id] = self.frame_count
                events.append(ReIDEvent(track_id, img_path, self.frame_count))

                # 6. Visual Hint: Flash Green Box over the YOLO box
                cv2.rectangle(frame, (det.x1, det.y1), (det.x2, det.y2), (0, 255, 0), 4)
                cv2.putText(frame, "SAVED", (det.x1, det.y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # ── Cleanup Memory ──
        # Remove tracks we haven't seen in a while (e.g., 60 frames) to prevent memory leaks
        stale_tracks = [
            tid for tid, last_frame in self.last_extracted.items() 
            if (self.frame_count - last_frame > 60) and (tid not in current_track_ids)
        ]
        for tid in stale_tracks:
            del self.last_extracted[tid]
        context["data"]["use_case"]["reid_events"] = [e.__dict__ for e in events]
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