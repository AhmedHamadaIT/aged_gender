"""
services/person_search.py
-----------------------------
PERSON_SEARCH task — Background Person Indexer with Progressive Overwrite.

Extracts appearance embeddings using OSNet and registers them in Qdrant.
To avoid redundancy, it only saves 1 image per track. It monitors the person's 
bounding box area and overwrites the saved crop/vector only when the person gets 
closer to the camera (producing a higher-resolution crop).

Task config shape (from POST /api/tasks):
{
    "taskId"        : int,
    "taskName"      : str,
    "algorithmType" : "PERSON_SEARCH",
    "channelId"     : int,
    "enable"        : bool,
    "detailConfig"  : {
        "padding": 10
    },
    "validWeekday"  : List[str],
    "validStartTime": int,
    "validEndTime"  : int
}
"""

import os
import json
import hashlib
import time
from datetime import datetime, timezone
from typing import List, Optional

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

# ── Constants ─────────────────────────────────────────────────────────────────

_WEEKDAY_MAP = {
    "MONDAY": 0, "TUESDAY": 1, "WEDNESDAY": 2, "THURSDAY": 3,
    "FRIDAY": 4, "SATURDAY": 5, "SUNDAY": 6,
}

# ── Task ──────────────────────────────────────────────────────────────────────

class PersonSearchTask:

    def __init__(self, task_config: dict):
        self.task_id    = task_config["taskId"]
        self.task_name  = task_config["taskName"]
        self.channel_id = task_config["channelId"]
        self.enable     = task_config.get("enable", True)

        detail       = task_config.get("detailConfig", {})
        self.padding = detail.get("padding", int(os.getenv("REID_PADDING", "10")))

        # Schedule
        raw_days            = task_config.get("validWeekday", list(_WEEKDAY_MAP.keys()))
        self.valid_weekdays = {_WEEKDAY_MAP[d] for d in raw_days if d in _WEEKDAY_MAP}
        self.valid_start_ms = task_config.get("validStartTime", 0)
        self.valid_end_ms   = task_config.get("validEndTime", 86400000)

        # ReID Model Setup
        model_path  = os.getenv("REID_MODEL_PATH", "models/osnet_x1_0.pt")
        self.device = torch.device(os.getenv("DEVICE", "cpu"))
        self.model  = torch.jit.load(model_path, map_location=str(self.device))
        self.model.eval()

        self.transform = T.Compose([
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.identity_manager = IdentityManager()

        # State tracking: { track_id: {"best_area": int, "last_frame": int} }
        self.track_state = {}  

        # Storage paths
        self._gallery_dir = os.getenv("GALLERY_DIR", "/local/storage/gallery")
        self._events_dir  = os.getenv("EVENTS_DIR",  "/local/storage/events")
        os.makedirs(self._gallery_dir, exist_ok=True)
        os.makedirs(self._events_dir,  exist_ok=True)

        self._jsonl_path = os.path.join(self._events_dir, f"task_{self.task_id}.jsonl")

        log.info(f"[PersonSearch/{self.task_id}] Indexer Ready — Using Progressive Overwrite")

    # ── Main Entry Point ──────────────────────────────────────────────────────

    def __call__(self, payload: dict) -> list:
        if not self.enable or not self._in_schedule():
            return []

        frame      = payload["frame"]
        frame_id   = payload["frame_id"]
        timestamp  = payload["timestamp"]
        detections = payload["detection"].get("items", [])

        # Filter strictly to tracked persons
        persons = [
            d for d in detections
            if d.class_name == "person" and d.track_id != -1
        ]

        events = []
        current_track_ids = set()

        for det in persons:
            tid = det.track_id
            current_track_ids.add(tid)

            x1, y1, x2, y2 = det.bbox
            current_area = (x2 - x1) * (y2 - y1)

            # Get existing state for this track, or initialize it
            state = self.track_state.get(tid, {"best_area": 0, "last_frame": frame_id})
            previous_best = state["best_area"]

            # Always update the frame we last saw them
            state["last_frame"] = frame_id

            # Trigger overwrite IF it's a new track OR the crop area is 20% larger than the previous best
            if current_area > (previous_best * 1.2):
                
                # 1. Crop
                crop = self._crop_bbox(frame, x1, y1, x2, y2)
                if crop.size == 0:
                    self.track_state[tid] = state
                    continue
                
                # 2. Extract Embedding
                tensor_feat = self._extract_raw_tensor(crop)
                if tensor_feat is None:
                    self.track_state[tid] = state
                    continue
                
                final_embedding = F.normalize(tensor_feat, p=2, dim=0).cpu().numpy().tolist()

                # 3. Create Deterministic Event ID based ONLY on Task + Track
                # This guarantees that when this person gets closer, we overwrite the exact same file
                event_id = hashlib.md5(f"{self.task_id}_{tid}".encode()).hexdigest()

                # 4. Build standard event structure
                event = self._build_event(det, event_id, timestamp)
                img_path = event["evidence"]["captureImage"]

                # 5. Save crop locally (Overwrites previous file instantly)
                os.makedirs(os.path.dirname(img_path), exist_ok=True)
                cv2.imwrite(img_path, crop)

                # 6. Insert to Vector DB 
                metadata = {
                    "eventId"   : event_id,
                    "image_path": img_path, 
                    "track_id"  : tid,
                    "frame_id"  : frame_id,
                    "taskId"    : self.task_id
                }
                # NOTE: Ensure your IdentityManager uses payload metadata (like eventId) to UPSERT 
                # instead of just appending a new point, so Qdrant stays 1-to-1 with the track.
                self.identity_manager.register(final_embedding, payload=metadata)

                # 7. Persist event to JSONL & update state
                self._persist_event(event)
                state["best_area"] = current_area
                events.append(event)
            
            # Save state back
            self.track_state[tid] = state

        # ── Memory Cleanup ──
        # Remove stale tracks that haven't been seen in the last 60 frames
        stale_tracks = [
            t for t, s in self.track_state.items() 
            if (frame_id - s["last_frame"] > 60) and (t not in current_track_ids)
        ]
        for t in stale_tracks:
            del self.track_state[t]

        return events

    # ── Event Construction & Persistence ──────────────────────────────────────

    def _build_event(self, det, event_id: str, timestamp: str) -> dict:
        now_ms = int(time.time() * 1000)

        date_str     = datetime.now().strftime("%Y/%m/%d")
        capture_path = os.path.join(self._gallery_dir, date_str, f"{event_id}_appearance.jpg")

        x1, y1, x2, y2 = det.bbox

        return {
            "eventId"     : event_id,
            "eventType"   : "PERSON_SEARCH",
            "timestamp"   : now_ms,
            "timestampUTC": datetime.fromtimestamp(
                now_ms / 1000, tz=timezone.utc
            ).isoformat().replace("+00:00", "Z"),
            "taskId"      : self.task_id,
            "taskName"    : self.task_name,
            "channelId"   : self.channel_id,
            "person": {
                "trackingId" : str(det.track_id),
                "boundingBox": {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1},
                "confidence" : int(det.confidence * 100)
            },
            "evidence": {
                "captureImage": capture_path
            }
        }

    def _persist_event(self, event: dict):
        """Append event payload directly to the task's JSONL log."""
        with open(self._jsonl_path, "a") as f:
            f.write(json.dumps(event) + "\n")

    # ── Inference & Image Utils ───────────────────────────────────────────────

    def _crop_bbox(self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1 - self.padding), max(0, y1 - self.padding)
        x2, y2 = min(w, x2 + self.padding), min(h, y2 + self.padding)
        return frame[y1:y2, x1:x2]

    def _extract_raw_tensor(self, cropped_bgr: np.ndarray) -> Optional[torch.Tensor]:
        """Runs the OSNet forward pass and returns un-normalized tensor features."""
        try:
            img_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
            tensor  = self.transform(Image.fromarray(img_rgb)).unsqueeze(0).to(self.device)

            with torch.no_grad():
                feature = self.model(tensor)[0]
            return feature
        except Exception as e:
            log.warning(f"[PersonSearch] Feature extraction failed: {e}")
            return None

    def search_by_image(self, image_bytes: bytes, top_k: int = 10) -> list[dict]:
        """
        Maintains backward compatibility for direct API queries if the route calls the task instance.
        """
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            log.error("[PersonSearch] Failed to decode image bytes from API request.")
            return []

        tensor_feat = self._extract_raw_tensor(img)
        if tensor_feat is None:
            return []

        query_vector = F.normalize(tensor_feat, p=2, dim=0).cpu().numpy().tolist()
        return self.identity_manager.search(query_vector, limit=top_k)

    # ── Schedule ──────────────────────────────────────────────────────────────

    def _in_schedule(self) -> bool:
        now = datetime.now()
        if now.weekday() not in self.valid_weekdays:
            return False
        ms_now = (now.hour * 3600 + now.minute * 60 + now.second) * 1000
        return self.valid_start_ms <= ms_now <= self.valid_end_ms