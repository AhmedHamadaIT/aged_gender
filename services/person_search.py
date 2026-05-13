"""
services/person_search.py
-----------------------------
Person Search service — OSNet-based appearance embedding extraction and search.

Provides:
  - extract_embedding(crop_bgr) → normalized embedding list (used by EmbeddingWorker)
  - search_by_image(image_bytes) → Qdrant search results (used by API endpoint)

Backends:
  - REID_MODEL_ONNX: optional OSNet-compatible ONNX (GPU-first via shared onnxruntime)
  - REID_MODEL_PATH: TorchScript .pt (default), uses DEVICE
"""

import os
from typing import Any, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from dotenv import load_dotenv

from logger.logger_config import Logger
from store.identity_manager import IdentityManager
from utils.ml_backend import (
    require_gpu_device_if_configured,
    resolve_ultralytics_device,
)

load_dotenv()
log = Logger.get_logger(__name__)


class PersonSearchService:

    def __init__(self):
        # ── OSNet: ONNX (preferred for GPU-ORT) or TorchScript ─────────────
        try:
            self.transform = T.Compose([
                T.Resize((256, 128)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self.identity_manager = IdentityManager()
            self._onnx: Any = None
            self._onnx_in_name: Optional[str] = None
            self.model: Any = None
            self.device: Any = None

            onnx_path = os.getenv("REID_MODEL_ONNX", "").strip()
            if onnx_path and os.path.exists(onnx_path):
                from utils.onnx_runtime import create_inference_session

                self._onnx = create_inference_session(onnx_path)
                self._onnx_in_name = self._onnx.get_inputs()[0].name
                self.model = None
                self.device = None
                log.info(
                    "[PersonSearchService] Ready — Re-ID ONNX: %s | providers=%s",
                    onnx_path,
                    self._onnx.get_providers(),
                )
            else:
                model_path = os.getenv("REID_MODEL_PATH", "models/osnet_x1_0.pt")
                if not os.path.exists(model_path):
                    raise ValueError(
                        f"The provided filename {model_path} does not exist "
                        f"(or set REID_MODEL_ONNX to a valid OSNet export)"
                    )
                _d = resolve_ultralytics_device()
                if isinstance(_d, int):
                    self.device = torch.device(f"cuda:{_d}")
                else:
                    self.device = torch.device(str(_d))
                self.model = torch.jit.load(
                    model_path, map_location=str(self.device)
                )
                self.model.eval()
                require_gpu_device_if_configured(_d, "PersonSearchService")
                log.info(
                    "[PersonSearchService] Ready — OSNet TorchScript: %s | device=%s",
                    model_path,
                    self.device,
                )
        except Exception as e:  # noqa: BLE001
            log.warning(
                "[PersonSearchService] Model not loaded — %s. Search endpoint will return 503.",
                e,
            )
            self.model = None
            self._onnx = None
            self.identity_manager = None

    def is_ready(self) -> bool:
        if self.identity_manager is None:
            return False
        return self._onnx is not None or self.model is not None

    # ── Embedding Extraction ──────────────────────────────────────────────

    def extract_embedding(self, crop_bgr: np.ndarray) -> Optional[list]:
        """
        Extract a normalized ReID embedding from a BGR crop.
        Returns a float list suitable for Qdrant, or None on failure.
        """
        if self._onnx is not None and self._onnx_in_name is not None:
            return self._extract_onnx(crop_bgr)
        if self.model is not None and self.device is not None:
            return self._extract_torch(crop_bgr)
        return None

    def _extract_onnx(self, crop_bgr: np.ndarray) -> Optional[list]:
        try:
            img_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            t = self.transform(Image.fromarray(img_rgb))
            np_input = t.unsqueeze(0).cpu().numpy().astype(np.float32)
            outs = self._onnx.run(None, {self._onnx_in_name: np_input})
            feat = np.array(outs[0], dtype=np.float32)
            if feat.ndim > 1:
                feat = feat[0]
            n = float(np.linalg.norm(feat) + 1e-8)
            return (feat / n).tolist()
        except Exception as e:  # noqa: BLE001
            log.warning("[PersonSearchService] ONNX feature extraction failed: %s", e)
            return None

    def _extract_torch(self, crop_bgr: np.ndarray) -> Optional[list]:
        try:
            img_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            tensor = self.transform(Image.fromarray(img_rgb)).unsqueeze(0).to(
                self.device
            )
            with torch.no_grad():
                feature = self.model(tensor)[0]
            return F.normalize(feature, p=2, dim=0).cpu().numpy().tolist()
        except Exception as e:  # noqa: BLE001
            log.warning(f"[PersonSearchService] Feature extraction failed: {e}")
            return None

    # ── Search (Query) ────────────────────────────────────────────────────

    def search_by_image(self, image_bytes: bytes, top_k: int = 10) -> list[dict]:
        """
        Decode image bytes, extract embedding, and search Qdrant.
        Used by the API endpoint for querying.
        """
        if not self.is_ready():
            raise RuntimeError(
                "Re-ID not loaded — set REID_MODEL_ONNX or REID_MODEL_PATH."
            )

        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            log.error("[PersonSearchService] Failed to decode image bytes.")
            return []

        embedding = self.extract_embedding(img)
        if embedding is None:
            return []

        return self.identity_manager.search(embedding, limit=top_k)
