"""
embedding_worker.py
-------------------
Background EmbeddingWorker — runs in its own process, decoupled from detection.

Consumes crop messages from the embedding_queue, extracts embeddings, and
batches Qdrant upserts to reduce latency spikes.
"""

import os
import hashlib
import pickle
import queue as _queue
import sqlite3
import time
from typing import Any, List, Tuple

import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv()

_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "8"))
_FLUSH_TIMEOUT = float(os.getenv("EMBED_FLUSH_TIMEOUT", "3.0"))
_DLQ_PATH = os.getenv("EMBED_DLQ_DB", os.path.join("artifacts", "embed_dlq.db"))
_DLQ_RETRY_INTERVAL = float(os.getenv("DLQ_RETRY_INTERVAL_SEC", "60"))
_DLQ_MAX_RETRIES = max(1, int(os.getenv("DLQ_MAX_RETRIES", "10")))


def _dlq_init() -> None:
    d = os.path.dirname(os.path.abspath(_DLQ_PATH))
    if d:
        os.makedirs(d, exist_ok=True)
    conn = sqlite3.connect(_DLQ_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS embed_dlq (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            kind TEXT NOT NULL,
            payload BLOB NOT NULL,
            retry_count INTEGER DEFAULT 0,
            created REAL
        )
        """
    )
    conn.commit()
    conn.close()


def _dlq_enqueue(kind: str, batch: List[Tuple[str, Any, dict]], log) -> None:
    if not batch:
        return
    try:
        conn = sqlite3.connect(_DLQ_PATH)
        conn.execute(
            "INSERT INTO embed_dlq (kind, payload, retry_count, created) VALUES (?,?,0,?)",
            (kind, sqlite3.Binary(pickle.dumps(batch, protocol=pickle.HIGHEST_PROTOCOL)), time.time()),
        )
        conn.commit()
        conn.close()
        log.warning("[EmbeddingWorker] queued %d items to DLQ kind=%s", len(batch), kind)
    except Exception as exc:
        log.warning("[EmbeddingWorker] DLQ enqueue failed: %s", exc)


def _dlq_retry_batch(
    person_search,
    semantic_search,
    person_search_ready: bool,
    semantic_search_ready: bool,
    log,
) -> int:
    """Replay up to one row from DLQ. Returns number successfully flushed."""
    try:
        conn = sqlite3.connect(_DLQ_PATH)
    except Exception:
        return 0
    try:
        cur = conn.execute(
            "SELECT id, kind, payload, retry_count FROM embed_dlq ORDER BY id LIMIT 1"
        )
        row = cur.fetchone()
        if not row:
            conn.close()
            return 0
        rid, kind, payload, rcount = row
        batch: List[Tuple[str, Any, dict]] = pickle.loads(payload)
        try:
            if kind == "reid" and person_search_ready:
                person_search.identity_manager.upsert_batch(batch, wait=False)
            elif kind == "semantic" and semantic_search_ready:
                semantic_search.identity_manager.upsert_batch(batch, wait=False)
            else:
                conn.close()
                return 0
            conn.execute("DELETE FROM embed_dlq WHERE id=?", (rid,))
            conn.commit()
            conn.close()
            log.info("[EmbeddingWorker] DLQ replay ok id=%s kind=%s n=%d", rid, kind, len(batch))
            return len(batch)
        except Exception as exc:
            r2 = int(rcount) + 1
            if r2 >= _DLQ_MAX_RETRIES:
                conn.execute("DELETE FROM embed_dlq WHERE id=?", (rid,))
                log.error(
                    "[EmbeddingWorker] DLQ discarded id=%s kind=%s after %s retries: %s",
                    rid,
                    kind,
                    r2,
                    exc,
                )
            else:
                conn.execute(
                    "UPDATE embed_dlq SET retry_count=? WHERE id=?",
                    (r2, rid),
                )
                log.warning("[EmbeddingWorker] DLQ retry failed id=%s: %s", rid, exc)
            conn.commit()
            conn.close()
            return 0
    except Exception as exc:
        log.warning("[EmbeddingWorker] DLQ retry error: %s", exc)
        try:
            conn.close()
        except Exception:
            pass
        return 0


def _embedding_is_valid(vec: Any) -> bool:
    if vec is None:
        return False
    arr = np.asarray(vec, dtype=np.float32)
    if arr.size == 0:
        return False
    if not np.isfinite(arr).all():
        return False
    if np.allclose(arr, 0, atol=1e-6):
        return False
    return True


def run_embedding_worker(embedding_queue, stop_event):
    from services.person_search import PersonSearchService
    from services.semantic_search import SemanticSearchService
    from logger.logger_config import Logger

    log = Logger.get_logger("EmbeddingWorker")
    _dlq_init()
    log.info("[EmbeddingWorker] Starting up — loading services...")

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

    log.info("[EmbeddingWorker] Ready — waiting for crops.")

    reid_batch: List[Tuple[str, Any, dict]] = []
    semantic_batch: List[Tuple[str, Any, dict]] = []
    last_flush = time.time()
    last_dlq_retry = time.time()
    processed = 0
    skipped = 0

    def _flush_reid():
        nonlocal reid_batch
        if not reid_batch:
            return
        try:
            person_search.identity_manager.upsert_batch(reid_batch, wait=False)
            log.debug(f"[EmbeddingWorker] ReID batch upserted: {len(reid_batch)} points")
        except Exception as e:
            log.warning(f"[EmbeddingWorker] ReID batch upsert failed: {e}")
            _dlq_enqueue("reid", list(reid_batch), log)
        reid_batch = []

    def _flush_semantic():
        nonlocal semantic_batch
        if not semantic_batch:
            return
        try:
            semantic_search.identity_manager.upsert_batch(semantic_batch, wait=False)
            log.debug(f"[EmbeddingWorker] CLIP batch upserted: {len(semantic_batch)} points")
        except Exception as e:
            log.warning(f"[EmbeddingWorker] CLIP batch upsert failed: {e}")
            _dlq_enqueue("semantic", list(semantic_batch), log)
        semantic_batch = []

    def _flush_all():
        nonlocal last_flush
        _flush_reid()
        _flush_semantic()
        last_flush = time.time()

    while not stop_event.is_set():
        if time.time() - last_flush >= _FLUSH_TIMEOUT:
            _flush_all()

        if time.time() - last_dlq_retry >= _DLQ_RETRY_INTERVAL:
            last_dlq_retry = time.time()
            while _dlq_retry_batch(
                person_search,
                semantic_search,
                person_search_ready,
                semantic_search_ready,
                log,
            ):
                pass

        try:
            msg = embedding_queue.get(timeout=1.0)
        except _queue.Empty:
            continue
        except Exception:
            continue

        camera_id = msg["camera_id"]
        track_id = msg["track_id"]
        crop_path = msg["crop_path"]
        frame_id = msg.get("frame_id", 0)
        confidence = msg.get("confidence", 0.0)
        timestamp = msg.get("timestamp", "")

        crop = cv2.imread(crop_path)
        if crop is None:
            log.warning(f"[EmbeddingWorker] Could not read crop: {crop_path}")
            skipped += 1
            continue

        if person_search_ready:
            reid_embedding = person_search.extract_embedding(crop)
            if _embedding_is_valid(reid_embedding):
                point_id = hashlib.md5(
                    f"reid_{camera_id}_{track_id}".encode()
                ).hexdigest()
                metadata = {
                    "type": "person_search",
                    "camera_id": camera_id,
                    "track_id": track_id,
                    "image_path": crop_path,
                    "frame_id": frame_id,
                    "confidence": confidence,
                    "timestamp": timestamp,
                }
                reid_batch.append((point_id, reid_embedding, metadata))
                if len(reid_batch) >= _BATCH_SIZE:
                    _flush_reid()
            else:
                log.debug(f"[EmbeddingWorker] skip reid track {track_id} — bad embedding")
                skipped += 1

        if semantic_search_ready:
            clip_embedding = semantic_search.extract_embedding(crop)
            if _embedding_is_valid(clip_embedding):
                point_id = hashlib.md5(
                    f"semantic_{camera_id}_{track_id}".encode()
                ).hexdigest()
                metadata = {
                    "type": "semantic_search",
                    "camera_id": camera_id,
                    "track_id": track_id,
                    "image_path": crop_path,
                    "frame_id": frame_id,
                    "confidence": confidence,
                    "timestamp": timestamp,
                }
                semantic_batch.append((point_id, clip_embedding, metadata))
                if len(semantic_batch) >= _BATCH_SIZE:
                    _flush_semantic()
            else:
                log.debug(f"[EmbeddingWorker] skip clip track {track_id} — bad embedding")
                skipped += 1

        processed += 1
        if processed % 50 == 0:
            total = processed + skipped
            pct = (skipped / total * 100) if total else 0.0
            log.info(
                f"[EmbeddingWorker] processed={processed} skipped={skipped} "
                f"skip_rate={pct:.1f}%"
            )

    _flush_all()
    log.info(f"[EmbeddingWorker] Stopped. processed={processed} skipped={skipped}")
