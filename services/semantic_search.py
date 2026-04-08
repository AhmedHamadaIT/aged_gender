"""
services/semantic_search.py
---------------------------
SEMANTIC_SEARCH task — Background Text-to-Image Indexer.

Extracts image embeddings using MobileCLIP2 via ONNX Runtime and registers them in Qdrant.
Maintains a 1-to-1 relationship between a tracked person and their Qdrant vector
by employing the Progressive Overwrite pattern (only updating when the bounding 
box gets significantly larger/closer).
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
from PIL import Image
from dotenv import load_dotenv

import open_clip
import onnxruntime as ort

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

class SemanticSearchTask:

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

        # ── Model & Transforms Setup ──
        self.device_str = os.getenv("DEVICE", "cuda")
        model_name  = os.getenv("MOBILECLIP_MODEL", "MobileCLIP2-S0")
        pretrained  = os.getenv("MOBILECLIP_PRETRAINED", "dfndr2b")
        
        log.info(f"[SemanticSearch/{self.task_id}] Loading Transforms for {model_name}...")
        
        # Load PyTorch model temporarily JUST to get the correct preprocess and tokenizer
        base_model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device="cpu"
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        
        # Free memory: we don't need the PyTorch model anymore
        del base_model 
        import gc; gc.collect()

        # ── ONNX Runtime Setup ──
        image_onnx_path = os.getenv("IMAGE_ENCODER_ONNX", "mobileclip2_image.onnx")
        text_onnx_path  = os.getenv("TEXT_ENCODER_ONNX", "mobileclip2_text.onnx")

        # Prioritize GPU if requested
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'cuda' in self.device_str else ['CPUExecutionProvider']

        log.info(f"[SemanticSearch/{self.task_id}] Initializing ONNX Sessions (Providers: {providers[0]})")
        
        self.image_session = ort.InferenceSession(image_onnx_path, providers=providers)
        self.text_session  = ort.InferenceSession(text_onnx_path, providers=providers)

        # Cache input names for the `run` method
        self.image_input_name = self.image_session.get_inputs()[0].name
        self.text_input_name  = self.text_session.get_inputs()[0].name

        self.identity_manager = IdentityManager()

        # State tracking: { track_id: {"best_area": int, "last_frame": int} }
        self.track_state = {}  

        # Storage paths
        self._gallery_dir = os.getenv("GALLERY_DIR", "/local/storage/gallery")
        self._events_dir  = os.getenv("EVENTS_DIR",  "/local/storage/events")
        os.makedirs(self._gallery_dir, exist_ok=True)
        os.makedirs(self._events_dir,  exist_ok=True)

        self._jsonl_path = os.path.join(self._events_dir, f"task_{self.task_id}.jsonl")

        log.info(f"[SemanticSearch/{self.task_id}] ONNX Indexer Ready")

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

            state = self.track_state.get(tid, {"best_area": 0, "last_frame": frame_id})
            previous_best = state["best_area"]
            state["last_frame"] = frame_id

            # Trigger overwrite IF it's a new track OR crop area is 20% larger than previous best
            if current_area > (previous_best * 1.2):
                
                # 1. Crop
                crop = self._crop_bbox(frame, x1, y1, x2, y2)
                if crop.size == 0:
                    self.track_state[tid] = state
                    continue
                
                # 2. Extract Embedding via ONNX
                np_feat = self._extract_image_features(crop)
                if np_feat is None:
                    self.track_state[tid] = state
                    continue
                
                # Normalize using Pure NumPy (equivalent to F.normalize)
                norm = np.linalg.norm(np_feat, axis=-1, keepdims=True)
                final_embedding = (np_feat / norm).squeeze().tolist()

                # 3. Create Deterministic Event ID
                event_id = hashlib.md5(f"{self.task_id}_{tid}".encode()).hexdigest()

                # 4. Build standard event structure
                event = self._build_event(det, event_id, timestamp)
                img_path = event["evidence"]["captureImage"]

                # 5. Save crop locally
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
                self.identity_manager.register(final_embedding, payload=metadata)

                # 7. Persist event to JSONL & update state
                self._persist_event(event)
                state["best_area"] = current_area
                events.append(event)
            
            self.track_state[tid] = state

        # ── Memory Cleanup ──
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
        date_str = datetime.now().strftime("%Y/%m/%d")
        capture_path = os.path.join(self._gallery_dir, date_str, f"{event_id}_semantic.jpg")

        x1, y1, x2, y2 = det.bbox

        return {
            "eventId"     : event_id,
            "eventType"   : "SEMANTIC_SEARCH",
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
        with open(self._jsonl_path, "a") as f:
            f.write(json.dumps(event) + "\n")

    # ── Inference & Image Utils ───────────────────────────────────────────────

    def _crop_bbox(self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1 - self.padding), max(0, y1 - self.padding)
        x2, y2 = min(w, x2 + self.padding), min(h, y2 + self.padding)
        return frame[y1:y2, x1:x2]

    def _extract_image_features(self, cropped_bgr: np.ndarray) -> Optional[np.ndarray]:
        try:
            img_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            
            # Preprocess creates a PyTorch tensor, we convert it to NumPy for ONNX
            tensor = self.preprocess(pil_img).unsqueeze(0)
            np_input = tensor.cpu().numpy().astype(np.float32)

            # Execute ONNX Session
            ort_outs = self.image_session.run(None, {self.image_input_name: np_input})
            return ort_outs[0]
            
        except Exception as e:
            log.warning(f"[SemanticSearch] Image feature extraction failed: {e}")
            return None

    # ── Dashboard API Hooks ───────────────────────────────────────────────────

    def search_by_image(self, image_bytes: bytes, top_k: int = 10) -> list[dict]:
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            log.error("[SemanticSearch] Failed to decode image bytes.")
            return []

        np_feat = self._extract_image_features(img)
        if np_feat is None:
            return []

        norm = np.linalg.norm(np_feat, axis=-1, keepdims=True)
        query_vector = (np_feat / norm).squeeze().tolist()
        return self.identity_manager.search(query_vector, limit=top_k)

    def search_by_text(self, text_query: str, top_k: int = 10) -> list[dict]:
        try:
            # Tokenizer creates a PyTorch tensor, convert to int64 NumPy array for ONNX
            tokens = self.tokenizer([text_query])
            np_input = tokens.cpu().numpy().astype(np.int64)
            
            # Execute ONNX Session
            ort_outs = self.text_session.run(None, {self.text_input_name: np_input})
            np_feat = ort_outs[0]

            norm = np.linalg.norm(np_feat, axis=-1, keepdims=True)
            query_vector = (np_feat / norm).squeeze().tolist()
            
            return self.identity_manager.search(query_vector, limit=top_k)
        except Exception as e:
            log.error(f"[SemanticSearch] Text search failed: {e}")
            return []

    # ── Schedule ──────────────────────────────────────────────────────────────

    def _in_schedule(self) -> bool:
        now = datetime.now()
        if now.weekday() not in self.valid_weekdays:
            return False
        ms_now = (now.hour * 3600 + now.minute * 60 + now.second) * 1000
        return self.valid_start_ms <= ms_now <= self.valid_end_ms