"""
store/identity_manager.py
-------------------------
Person identity gallery backed by Qdrant vector database.

Provides similarity search (identify) and enrollment (register)
for ReID feature vectors. Used by services/feature_extractor.py (ReIDService).

Environment variables:
    QDRANT_URL            — Qdrant server URL       (default: http://localhost:6333)
    REID_COLLECTION       — Collection name          (default: reid_gallery)
    REID_MATCH_THRESHOLD  — Cosine similarity cutoff (default: 0.75)
    REID_VECTOR_SIZE      — Embedding dimensionality (default: 512)
"""

import os
import uuid
from typing import Optional, List, Dict

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance


from logger.logger_config import Logger

load_dotenv()
log = Logger.get_logger(__name__)


class IdentityManager:
    def __init__(self):
        qdrant_url      = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.collection  = os.getenv("REID_COLLECTION", "reid_gallery")
        self.threshold   = float(os.getenv("REID_MATCH_THRESHOLD", "0.75"))
        vector_size      = int(os.getenv("REID_VECTOR_SIZE", "512"))

        log.info(f"[IDENTITY] Connecting to Qdrant: {qdrant_url}")
        self.client = QdrantClient(url=qdrant_url)

        # Ensure collection exists
        try:
            if not self.client.collection_exists(self.collection):
                self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE,
                    ),
                )
                log.info(
                    f"[IDENTITY] Created collection '{self.collection}' "
                    f"(dim={vector_size}, cosine)"
                )
            else:
                log.info(f"[IDENTITY] Collection '{self.collection}' already exists")
        except Exception as e:
            log.error(f"[IDENTITY] Failed to initialize Qdrant collection: {e}")
            raise

        log.info(
            f"[IDENTITY] Ready — threshold={self.threshold}, "
            f"vector_size={vector_size}\n"
        )

    # ── Public API ────────────────────────────

    def search(self, feature_vector: list[float], limit: int = 10) -> List[Dict]:
        """
        Query the gallery for the top N closest matches to build the UI gallery.
        """
        try:
            hits = self.client.query_points(
                collection_name=self.collection,
                query=feature_vector,
                limit=limit,
            ).points
            
            return [
                {
                    "score": round(hit.score, 4),
                    "metadata": hit.payload
                }
                for hit in hits
            ]
        except Exception as e:
            # log.error(f"[IDENTITY] Qdrant search failed: {e}")
            return []

    def register(self, feature_vector: list[float], payload: dict = None) -> str:
        """
        Save the extracted embedding and metadata (image_path, frame, etc.) to the gallery.
        """
        point_id = str(uuid.uuid4())
        payload = payload or {}

        try:
            self.client.upsert(
                collection_name=self.collection,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=feature_vector,
                        payload=payload,
                    )
                ],
            )
            # log.info(f"[IDENTITY] Registered new vector with payload: {payload}")
        except Exception as e:
            # log.error(f"[IDENTITY] Failed to register vector: {e}")
            raise

        return point_id
