"""
embedding_worker.py
-------------------
Background EmbeddingWorker — runs in its own process, fully decoupled
from the detection pipeline.

Consumes crop messages from the embedding_queue (emitted by FrameBus),
loads the crop image from disk, and uses PersonSearchService and
SemanticSearchService to extract embeddings and upsert into Qdrant.

Uses deterministic Qdrant point IDs so that progressive overwrites
replace old vectors rather than appending duplicates.
"""

import os
import hashlib
import queue as _queue

import cv2
from dotenv import load_dotenv

load_dotenv()


def run_embedding_worker(embedding_queue, stop_event):
    """
    Entry point for the EmbeddingWorker process.

    Args:
        embedding_queue : Input queue — receives crop messages from FrameBus.
        stop_event      : Shared event; set when cameras are stopping.
    """
    from services.person_search import PersonSearchService
    from services.semantic_search import SemanticSearchService
    from logger.logger_config import Logger

    log = Logger.get_logger("EmbeddingWorker")
    log.info("[EmbeddingWorker] Starting up — loading services...")

    # ── Initialize services (models loaded once here) ─────────────────────
    person_search = PersonSearchService()
    semantic_search = SemanticSearchService()

    person_search_ready = (
        getattr(person_search, "model", None) is not None
        and getattr(person_search, "identity_manager", None) is not None
    )
    semantic_search_ready = (
        getattr(semantic_search, "_ready", False)
        and getattr(semantic_search, "identity_manager", None) is not None
    )

    if not person_search_ready:
        log.info("[EmbeddingWorker] Person search disabled — REID model unavailable.")
    if not semantic_search_ready:
        log.info("[EmbeddingWorker] Semantic search disabled — ONNX models unavailable.")

    log.info("[EmbeddingWorker] Ready — waiting for crops.\n")

    # ── Main Loop ─────────────────────────────────────────────────────────
    while not stop_event.is_set():
        try:
            msg = embedding_queue.get(timeout=1.0)
        except _queue.Empty:
            continue
        except Exception:
            continue

        camera_id  = msg["camera_id"]
        track_id   = msg["track_id"]
        crop_path  = msg["crop_path"]
        frame_id   = msg.get("frame_id", 0)
        confidence = msg.get("confidence", 0.0)
        timestamp  = msg.get("timestamp", "")

        # Read crop from disk
        crop = cv2.imread(crop_path)
        if crop is None:
            log.warning(f"[EmbeddingWorker] Could not read crop: {crop_path}")
            continue

        # ── 1. Person Search (ReID) Embedding ─────────────────────────────
        reid_embedding = person_search.extract_embedding(crop) if person_search_ready else None
        if reid_embedding is not None:
            point_id = hashlib.md5(f"reid_{camera_id}_{track_id}".encode()).hexdigest()
            metadata = {
                "type"       : "person_search",
                "camera_id"  : camera_id,
                "track_id"   : track_id,
                "image_path" : crop_path,
                "frame_id"   : frame_id,
                "confidence" : confidence,
                "timestamp"  : timestamp,
            }
            try:
                person_search.identity_manager.upsert_by_id(
                    point_id, reid_embedding, payload=metadata
                )
            except Exception as e:
                log.warning(f"[EmbeddingWorker] ReID upsert failed: {e}")

        # ── 2. Semantic Search (CLIP) Embedding ───────────────────────────
        clip_embedding = semantic_search.extract_embedding(crop) if semantic_search_ready else None
        if clip_embedding is not None:
            point_id = hashlib.md5(f"semantic_{camera_id}_{track_id}".encode()).hexdigest()
            metadata = {
                "type"       : "semantic_search",
                "camera_id"  : camera_id,
                "track_id"   : track_id,
                "image_path" : crop_path,
                "frame_id"   : frame_id,
                "confidence" : confidence,
                "timestamp"  : timestamp,
            }
            try:
                semantic_search.identity_manager.upsert_by_id(
                    point_id, clip_embedding, payload=metadata
                )
            except Exception as e:
                log.warning(f"[EmbeddingWorker] CLIP upsert failed: {e}")

    log.info("[EmbeddingWorker] Stopped.")
