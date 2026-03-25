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
from typing import Optional

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

    def identify(self, feature_vector: list[float]) -> Optional[dict]:
        """
        Search the gallery for the closest match.

        Returns:
            {"person_id": str, "confidence": float}  if score >= threshold
            None                                      if no match found
        """
        try:
            hits = self.client.search(
                collection_name=self.collection,
                query_vector=feature_vector,
                limit=1,
            )
        except Exception as e:
            log.error(f"[IDENTITY] Qdrant search failed: {e}")
            return None

        if hits and hits[0].score >= self.threshold:
            return {
                "person_id":  hits[0].payload["person_id"],
                "confidence": round(hits[0].score, 4),
            }

        return None

    def register(self, feature_vector: list[float], person_id: str = None) -> str:
        """
        Enroll a new person into the gallery.

        Args:
            feature_vector: Normalized embedding vector.
            person_id:      Optional explicit ID. If None, a UUID is generated.

        Returns:
            The person_id that was stored.
        """
        if person_id is None:
            person_id = str(uuid.uuid4())

        point_id = str(uuid.uuid4())   # Qdrant point ID (unique per vector)

        try:
            self.client.upsert(
                collection_name=self.collection,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=feature_vector,
                        payload={"person_id": person_id},
                    )
                ],
            )
            log.info(f"[IDENTITY] Registered new person: {person_id}")
        except Exception as e:
            log.error(f"[IDENTITY] Failed to register person: {e}")
            raise

        return person_id
