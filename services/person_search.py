"""
services/person_search.py
-----------------------------
Person Search service — OSNet-based appearance embedding extraction and search.

Provides:
  - extract_embedding(crop_bgr) → normalized embedding list (used by EmbeddingWorker)
  - search_by_image(image_bytes) → Qdrant search results (used by API endpoint)
"""

import os
from typing import Optional

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


class PersonSearchService:

    def __init__(self):
        # ── OSNet Model Setup ─────────────────────────────────────────────
        try:
            model_path = os.getenv("REID_MODEL_PATH", "models/osnet_x1_0.pt")
            if not os.path.exists(model_path):
                raise ValueError(f"The provided filename {model_path} does not exist")
            self.device = torch.device(os.getenv("DEVICE", "cpu"))
            self.model  = torch.jit.load(model_path, map_location=str(self.device))
            self.model.eval()

            self.transform = T.Compose([
                T.Resize((256, 128)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            self.identity_manager = IdentityManager()
            log.info("[PersonSearchService] Ready — OSNet loaded")
        except Exception as e:
            log.warning(f"[PersonSearchService] Model not loaded — {e}. Search endpoint will return 503.")
            self.model = None
            self.identity_manager = None

    # ── Embedding Extraction ──────────────────────────────────────────────

    def extract_embedding(self, crop_bgr: np.ndarray) -> Optional[list]:
        """
        Extract a normalized ReID embedding from a BGR crop.
        Returns a float list suitable for Qdrant, or None on failure.
        """
        try:
            img_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            tensor = self.transform(Image.fromarray(img_rgb)).unsqueeze(0).to(self.device)

            with torch.no_grad():
                feature = self.model(tensor)[0]

            return F.normalize(feature, p=2, dim=0).cpu().numpy().tolist()
        except Exception as e:
            log.warning(f"[PersonSearchService] Feature extraction failed: {e}")
            return None

    # ── Search (Query) ────────────────────────────────────────────────────

    def search_by_image(self, image_bytes: bytes, top_k: int = 10) -> list[dict]:
        """
        Decode image bytes, extract embedding, and search Qdrant.
        Used by the API endpoint for querying.
        """
        if self.model is None:
            raise RuntimeError("OSNet model not loaded — REID_MODEL_PATH missing or invalid.")

        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            log.error("[PersonSearchService] Failed to decode image bytes.")
            return []

        embedding = self.extract_embedding(img)
        if embedding is None:
            return []

        return self.identity_manager.search(embedding, limit=top_k)