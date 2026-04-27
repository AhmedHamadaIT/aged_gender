"""
services/semantic_search.py
---------------------------
Semantic Search service — MobileCLIP-based image/text embedding extraction and search.

Provides:
  - extract_embedding(crop_bgr) → normalized image embedding list (used by EmbeddingWorker)
  - search_by_image(image_bytes) → Qdrant search results (used by API endpoint)
  - search_by_text(text_query)   → Qdrant search results (used by API endpoint)
"""

import os
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from dotenv import load_dotenv

from logger.logger_config import Logger
from store.identity_manager import IdentityManager

load_dotenv()
log = Logger.get_logger(__name__)


class SemanticSearchService:

    def __init__(self):
        self._ready = False
        self.identity_manager = None
        try:
            image_onnx_path = os.getenv("IMAGE_ENCODER_ONNX", "./models/image_encoder.onnx")
            text_onnx_path  = os.getenv("TEXT_ENCODER_ONNX", "./models/text_encoder.onnx")
            missing_paths = [p for p in (image_onnx_path, text_onnx_path) if not os.path.exists(p)]
            if missing_paths:
                raise FileNotFoundError(
                    f"Missing ONNX model(s): {', '.join(missing_paths)}"
                )

            import gc
            import open_clip

            from utils.onnx_runtime import create_inference_session

            # ── MobileCLIP Model + Transforms Setup ──────────────────────────
            model_name = os.getenv("MOBILECLIP_MODEL", "MobileCLIP2-S0")
            pretrained = os.getenv("MOBILECLIP_PRETRAINED", "dfndr2b")

            log.info(f"[SemanticSearchService] Loading transforms for {model_name}...")

            # Load PyTorch model temporarily JUST to get the correct preprocess and tokenizer.
            base_model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained, device="cpu"
            )
            self.tokenizer = open_clip.get_tokenizer(model_name)

            del base_model
            gc.collect()

            # GPU-first (TensorRT → CUDA → CPU) — shared with other ONNX services
            self.image_session = create_inference_session(image_onnx_path)
            self.text_session = create_inference_session(text_onnx_path)

            self.image_input_name = self.image_session.get_inputs()[0].name
            self.text_input_name  = self.text_session.get_inputs()[0].name

            self.identity_manager = IdentityManager()
            self._ready = True
            log.info("[SemanticSearchService] Ready — ONNX loaded")
        except Exception as e:
            log.warning(f"[SemanticSearchService] Not loaded — {e}. Search endpoints will return 503.")

    # ── Embedding Extraction ──────────────────────────────────────────────

    def extract_embedding(self, crop_bgr: np.ndarray) -> Optional[list]:
        """
        Extract a normalized CLIP image embedding from a BGR crop.
        Returns a float list suitable for Qdrant, or None on failure.
        """
        if not self._ready:
            raise RuntimeError("SemanticSearchService not loaded — ONNX models missing or open_clip unavailable.")
        try:
            img_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)

            tensor = self.preprocess(pil_img).unsqueeze(0)
            np_input = tensor.cpu().numpy().astype(np.float32)

            ort_outs = self.image_session.run(None, {self.image_input_name: np_input})
            np_feat = ort_outs[0]

            norm = np.linalg.norm(np_feat, axis=-1, keepdims=True)
            return (np_feat / norm).squeeze().tolist()
        except Exception as e:
            log.warning(f"[SemanticSearchService] Image feature extraction failed: {e}")
            return None

    # ── Search (Query) ────────────────────────────────────────────────────

    def search_by_image(self, image_bytes: bytes, top_k: int = 10) -> list[dict]:
        """Decode image bytes, extract CLIP embedding, and search Qdrant."""
        if not self._ready:
            raise RuntimeError("SemanticSearchService not loaded — ONNX models missing or open_clip unavailable.")
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            log.error("[SemanticSearchService] Failed to decode image bytes.")
            return []

        embedding = self.extract_embedding(img)
        if embedding is None:
            return []

        return self.identity_manager.search(embedding, limit=top_k)

    def search_by_text(self, text_query: str, top_k: int = 10) -> list[dict]:
        """Tokenize text, extract CLIP text embedding, and search Qdrant."""
        if not self._ready:
            raise RuntimeError("SemanticSearchService not loaded — ONNX models missing or open_clip unavailable.")
        try:
            tokens = self.tokenizer([text_query])
            np_input = tokens.cpu().numpy().astype(np.int64)

            ort_outs = self.text_session.run(None, {self.text_input_name: np_input})
            np_feat = ort_outs[0]

            norm = np.linalg.norm(np_feat, axis=-1, keepdims=True)
            query_vector = (np_feat / norm).squeeze().tolist()

            return self.identity_manager.search(query_vector, limit=top_k)
        except Exception as e:
            log.error(f"[SemanticSearchService] Text search failed: {e}")
            return []