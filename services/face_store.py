"""
services/face_store.py
----------------------
FAISS-backed face embedding store with JSON metadata sidecar.

Supports:
    - Library CRUD (create / delete / list)
    - Person CRUD (add / delete / list with multi-image registration)
    - Cross-library search (lib_ids="-1" searches all)
    - Multi-embedding voting (max score per person across all their embeddings)
    - Stranger store (separate FAISS index for unknowns, auto-dedup)

Index type: IndexIDMap(IndexFlatIP)
    - IndexFlatIP: exhaustive inner-product search (cosine on L2-normalised vectors)
    - IndexIDMap: wraps it with custom int64 IDs, enabling remove_ids() for
      incremental person deletion without full index rebuild.

    FAISS ID scheme: person_id * 10000 + embedding_index_within_person
    This gives up to 10,000 embeddings per person (far more than needed).

Storage layout on disk:
    {FACE_STORAGE_DIR}/
    ├── libraries/
    │   └── lib_{id}/
    │       ├── index.faiss
    │       ├── metadata.json
    │       └── faces/
    │           └── person_{id}_{idx}.jpg
    └── surveillance/
        ├── strangers.faiss
        ├── strangers_meta.json
        └── faces/
"""

import json
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

_faiss = None


def _get_faiss():
    global _faiss
    if _faiss is None:
        import faiss
        _faiss = faiss
    return _faiss


# ─────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────
@dataclass
class MatchResult:
    person_id  : int
    person_name: str
    lib_id     : int
    score      : float          # similarity 0-100
    face_image : str = ""


@dataclass
class StrangerMatch:
    stranger_id : int
    score       : float
    first_seen  : str
    last_seen   : str
    occurrence  : int
    face_image  : str = ""


@dataclass
class PersonRecord:
    person_id       : int
    name            : str
    embedding_indices: List[int]
    face_images     : List[str]


@dataclass
class LibraryMeta:
    lib_id  : int
    name    : str
    persons : Dict[int, PersonRecord] = field(default_factory=dict)


# ─────────────────────────────────────────────
# FAISS ID helpers
# ─────────────────────────────────────────────
_EMBEDDINGS_PER_PERSON = 10000   # max embeddings per person (ID namespace)


def _make_faiss_id(person_id: int, emb_idx: int) -> int:
    """Deterministic FAISS ID: person_id * 10000 + emb_idx."""
    return person_id * _EMBEDDINGS_PER_PERSON + emb_idx


def _person_from_faiss_id(faiss_id: int) -> int:
    """Extract person_id from a FAISS ID."""
    return faiss_id // _EMBEDDINGS_PER_PERSON


# ─────────────────────────────────────────────
# Face Store
# ─────────────────────────────────────────────
class FaceStore:
    """
    Thread-safe FAISS-backed face embedding store.

    Uses IndexIDMap(IndexFlatIP) — inner-product on normalised vectors = cosine
    similarity, with custom int64 IDs for incremental add/remove.
    """

    EMBEDDING_DIM = 512

    def __init__(self, storage_dir: str = None):
        self._storage_dir = Path(storage_dir or os.getenv("FACE_STORAGE_DIR", "./data/face"))
        self._lib_dir     = self._storage_dir / "libraries"
        self._surv_dir    = self._storage_dir / "surveillance"
        self._lock        = threading.RLock()

        self._libraries : Dict[int, LibraryMeta] = {}
        self._indices   : Dict[int, object]      = {}   # lib_id → IndexIDMap

        # O(1) lookup: FAISS ID → person_id (per library)
        self._id_to_person : Dict[int, Dict[int, int]] = {}  # {lib_id: {faiss_id: person_id}}

        self._stranger_index   = None
        self._stranger_meta    : List[dict] = []
        self._stranger_next_id = 1

        self._lib_dir.mkdir(parents=True, exist_ok=True)
        self._surv_dir.mkdir(parents=True, exist_ok=True)
        (self._surv_dir / "faces").mkdir(exist_ok=True)

        self._load_all()
        print(f"[FaceStore] Loaded {len(self._libraries)} libraries, "
              f"{len(self._stranger_meta)} strangers from {self._storage_dir}")

    # ── Loading ───────────────────────────────

    def _load_all(self):
        faiss = _get_faiss()

        for lib_path in sorted(self._lib_dir.iterdir()) if self._lib_dir.exists() else []:
            if not lib_path.is_dir():
                continue
            meta_path  = lib_path / "metadata.json"
            index_path = lib_path / "index.faiss"
            if not meta_path.exists():
                continue
            try:
                raw = json.loads(meta_path.read_text())
                lib_id = raw["lib_id"]
                lib_meta = LibraryMeta(lib_id=lib_id, name=raw.get("name", f"Library {lib_id}"))
                id_map = {}
                for p in raw.get("persons", []):
                    pr = PersonRecord(
                        person_id=p["person_id"], name=p["name"],
                        embedding_indices=p.get("embedding_indices", []),
                        face_images=p.get("face_images", []),
                    )
                    lib_meta.persons[pr.person_id] = pr
                    for idx in pr.embedding_indices:
                        id_map[idx] = pr.person_id
                self._libraries[lib_id] = lib_meta
                self._id_to_person[lib_id] = id_map

                if index_path.exists():
                    self._indices[lib_id] = faiss.read_index(str(index_path))
                else:
                    self._indices[lib_id] = faiss.IndexIDMap(
                        faiss.IndexFlatIP(self.EMBEDDING_DIM))
            except Exception as e:
                print(f"[FaceStore] Failed to load {lib_path.name}: {e}")

        stranger_idx_path  = self._surv_dir / "strangers.faiss"
        stranger_meta_path = self._surv_dir / "strangers_meta.json"
        if stranger_idx_path.exists():
            try:
                self._stranger_index = faiss.read_index(str(stranger_idx_path))
            except Exception:
                self._stranger_index = faiss.IndexFlatIP(self.EMBEDDING_DIM)
        else:
            self._stranger_index = faiss.IndexFlatIP(self.EMBEDDING_DIM)

        if stranger_meta_path.exists():
            try:
                self._stranger_meta = json.loads(stranger_meta_path.read_text())
                if self._stranger_meta:
                    self._stranger_next_id = max(s["id"] for s in self._stranger_meta) + 1
            except Exception:
                self._stranger_meta = []

    # ── Persistence ───────────────────────────

    def _save_library(self, lib_id: int):
        faiss = _get_faiss()
        lib_path = self._lib_dir / f"lib_{lib_id}"
        lib_path.mkdir(parents=True, exist_ok=True)
        lib = self._libraries[lib_id]
        meta = {
            "lib_id": lib.lib_id, "name": lib.name,
            "persons": [
                {"person_id": p.person_id, "name": p.name,
                 "embedding_indices": p.embedding_indices, "face_images": p.face_images}
                for p in lib.persons.values()
            ],
        }
        (lib_path / "metadata.json").write_text(json.dumps(meta, indent=2))
        if lib_id in self._indices:
            faiss.write_index(self._indices[lib_id], str(lib_path / "index.faiss"))

    def _save_strangers(self):
        faiss = _get_faiss()
        if self._stranger_index is not None:
            faiss.write_index(self._stranger_index, str(self._surv_dir / "strangers.faiss"))
        (self._surv_dir / "strangers_meta.json").write_text(
            json.dumps(self._stranger_meta, indent=2))

    # ── Library CRUD ──────────────────────────

    def create_library(self, lib_id: int, name: str) -> dict:
        faiss = _get_faiss()
        with self._lock:
            if lib_id in self._libraries:
                raise ValueError(f"Library {lib_id} already exists")
            self._libraries[lib_id] = LibraryMeta(lib_id=lib_id, name=name)
            self._indices[lib_id]   = faiss.IndexIDMap(
                faiss.IndexFlatIP(self.EMBEDDING_DIM))
            self._id_to_person[lib_id] = {}
            lib_path = self._lib_dir / f"lib_{lib_id}"
            lib_path.mkdir(parents=True, exist_ok=True)
            (lib_path / "faces").mkdir(exist_ok=True)
            self._save_library(lib_id)
            return {"lib_id": lib_id, "name": name, "person_count": 0}

    def delete_library(self, lib_id: int) -> bool:
        with self._lock:
            if lib_id not in self._libraries:
                return False
            del self._libraries[lib_id]
            self._indices.pop(lib_id, None)
            self._id_to_person.pop(lib_id, None)
            lib_path = self._lib_dir / f"lib_{lib_id}"
            if lib_path.exists():
                shutil.rmtree(lib_path)
            return True

    def list_libraries(self) -> List[dict]:
        with self._lock:
            return [
                {"lib_id": m.lib_id, "name": m.name, "person_count": len(m.persons),
                 "embedding_count": self._indices[m.lib_id].ntotal
                 if m.lib_id in self._indices else 0}
                for m in self._libraries.values()
            ]

    def get_library(self, lib_id: int) -> Optional[dict]:
        with self._lock:
            m = self._libraries.get(lib_id)
            if m is None:
                return None
            return {
                "lib_id": m.lib_id, "name": m.name, "person_count": len(m.persons),
                "persons": [
                    {"person_id": p.person_id, "name": p.name, "face_count": len(p.face_images)}
                    for p in m.persons.values()
                ],
            }

    # ── Person CRUD ───────────────────────────

    def add_person(self, lib_id: int, person_id: int, name: str,
                   embeddings: List[np.ndarray],
                   face_images: List[np.ndarray] = None) -> dict:
        faiss = _get_faiss()
        with self._lock:
            if lib_id not in self._libraries:
                raise ValueError(f"Library {lib_id} not found")
            lib = self._libraries[lib_id]
            if person_id in lib.persons:
                raise ValueError(f"Person {person_id} already exists in library {lib_id}")

            index  = self._indices[lib_id]
            id_map = self._id_to_person[lib_id]
            saved_faces = []

            if face_images:
                faces_dir = self._lib_dir / f"lib_{lib_id}" / "faces"
                faces_dir.mkdir(parents=True, exist_ok=True)
                for i, img in enumerate(face_images):
                    fname = f"person_{person_id}_{i}.jpg"
                    cv2.imwrite(str(faces_dir / fname), img)
                    saved_faces.append(fname)

            emb_indices = []
            for i, emb in enumerate(embeddings):
                emb_np = np.array(emb, dtype=np.float32).reshape(1, -1)
                faiss.normalize_L2(emb_np)
                faiss_id = _make_faiss_id(person_id, i)
                ids = np.array([faiss_id], dtype=np.int64)
                index.add_with_ids(emb_np, ids)
                emb_indices.append(faiss_id)
                id_map[faiss_id] = person_id

            lib.persons[person_id] = PersonRecord(
                person_id=person_id, name=name,
                embedding_indices=emb_indices, face_images=saved_faces)
            self._save_library(lib_id)
            return {"person_id": person_id, "name": name, "lib_id": lib_id,
                    "embeddings": len(embeddings), "face_images": saved_faces}

    def delete_person(self, lib_id: int, person_id: int) -> bool:
        faiss = _get_faiss()
        with self._lock:
            if lib_id not in self._libraries:
                return False
            lib = self._libraries[lib_id]
            if person_id not in lib.persons:
                return False
            person = lib.persons.pop(person_id)

            # Remove face images from disk
            faces_dir = self._lib_dir / f"lib_{lib_id}" / "faces"
            for fname in person.face_images:
                fpath = faces_dir / fname
                if fpath.exists():
                    fpath.unlink()

            # Remove embeddings from FAISS index by ID (no rebuild needed)
            if person.embedding_indices and lib_id in self._indices:
                ids_to_remove = np.array(person.embedding_indices, dtype=np.int64)
                self._indices[lib_id].remove_ids(ids_to_remove)

            # Clean up the id_map
            id_map = self._id_to_person.get(lib_id, {})
            for idx in person.embedding_indices:
                id_map.pop(idx, None)

            self._save_library(lib_id)
            return True

    def list_persons(self, lib_id: int) -> Optional[List[dict]]:
        with self._lock:
            if lib_id not in self._libraries:
                return None
            return [
                {"person_id": p.person_id, "name": p.name,
                 "face_count": len(p.face_images), "embedding_count": len(p.embedding_indices)}
                for p in self._libraries[lib_id].persons.values()
            ]

    # ── Search ────────────────────────────────

    def search(self, embedding: np.ndarray, lib_ids: str = "-1",
               top_k: int = 1, threshold: float = 70.0) -> List[MatchResult]:
        """
        Search across libraries for the closest matching person(s).

        Multi-embedding voting: when a person has multiple enrolled embeddings,
        the best (highest) score across all their embeddings is used.
        This gives robust matching without averaging.
        """
        faiss = _get_faiss()
        with self._lock:
            query = np.array(embedding, dtype=np.float32).reshape(1, -1)
            faiss.normalize_L2(query)

            target_libs = (list(self._libraries.keys()) if lib_ids == "-1"
                           else [int(x.strip()) for x in lib_ids.split(",") if x.strip()])

            # Collect best score per (lib_id, person_id)
            person_scores: Dict[Tuple[int, int], float] = {}

            for lid in target_libs:
                if lid not in self._indices:
                    continue
                index = self._indices[lid]
                if index.ntotal == 0:
                    continue

                # Search for more results to allow voting across multiple embeddings
                k = min(top_k * 5, index.ntotal)
                scores, ids = index.search(query, k)
                id_map = self._id_to_person.get(lid, {})

                for i in range(k):
                    faiss_id = int(ids[0][i])
                    if faiss_id < 0:
                        continue
                    sim_score = float(scores[0][i]) * 100.0
                    if sim_score < threshold:
                        continue

                    # O(1) lookup: FAISS ID → person_id
                    pid = id_map.get(faiss_id)
                    if pid is None:
                        # Fallback: derive from FAISS ID scheme
                        pid = _person_from_faiss_id(faiss_id)

                    key = (lid, pid)
                    # Keep the best score for this person (max voting)
                    if key not in person_scores or sim_score > person_scores[key]:
                        person_scores[key] = sim_score

            # Build results sorted by score
            all_results = []
            for (lid, pid), score in person_scores.items():
                lib_meta = self._libraries.get(lid)
                if lib_meta is None:
                    continue
                person = lib_meta.persons.get(pid)
                if person is None:
                    continue
                all_results.append(MatchResult(
                    person_id=pid, person_name=person.name, lib_id=lid,
                    score=round(score, 1),
                    face_image=person.face_images[0] if person.face_images else ""))

            all_results.sort(key=lambda r: r.score, reverse=True)
            return all_results[:top_k]

    # ── Stranger management ───────────────────

    def add_stranger(self, embedding: np.ndarray,
                     face_image: np.ndarray = None, metadata: dict = None) -> int:
        faiss = _get_faiss()
        with self._lock:
            query = np.array(embedding, dtype=np.float32).reshape(1, -1)
            faiss.normalize_L2(query)

            # Check if already seen (cosine > 0.6)
            if self._stranger_index.ntotal > 0:
                scores, indices = self._stranger_index.search(query, 1)
                if scores[0][0] > 0.6 and indices[0][0] >= 0:
                    idx = int(indices[0][0])
                    if idx < len(self._stranger_meta):
                        now = time.strftime("%Y-%m-%dT%H:%M:%S")
                        self._stranger_meta[idx]["last_seen"] = (
                            metadata.get("timestamp", now) if metadata else now)
                        self._stranger_meta[idx]["occurrence"] += 1
                        self._save_strangers()
                        return self._stranger_meta[idx]["id"]

            sid = self._stranger_next_id
            self._stranger_next_id += 1
            self._stranger_index.add(query)

            now = time.strftime("%Y-%m-%dT%H:%M:%S")
            face_fname = ""
            if face_image is not None:
                face_fname = f"stranger_{sid}.jpg"
                cv2.imwrite(str(self._surv_dir / "faces" / face_fname), face_image)

            record = {
                "id": sid,
                "first_seen": metadata.get("timestamp", now) if metadata else now,
                "last_seen" : metadata.get("timestamp", now) if metadata else now,
                "occurrence": 1,
                "face_image": face_fname,
            }
            self._stranger_meta.append(record)
            self._save_strangers()
            return sid

    def search_strangers(self, embedding: np.ndarray,
                         top_k: int = 5, threshold: float = 30.0) -> List[StrangerMatch]:
        faiss = _get_faiss()
        with self._lock:
            if self._stranger_index.ntotal == 0:
                return []
            query = np.array(embedding, dtype=np.float32).reshape(1, -1)
            faiss.normalize_L2(query)
            k = min(top_k, self._stranger_index.ntotal)
            scores, indices = self._stranger_index.search(query, k)
            results = []
            for i in range(k):
                idx = int(indices[0][i])
                if idx < 0 or idx >= len(self._stranger_meta):
                    continue
                sim = float(scores[0][i]) * 100.0
                if sim < threshold:
                    continue
                rec = self._stranger_meta[idx]
                results.append(StrangerMatch(
                    stranger_id=rec["id"], score=round(sim, 1),
                    first_seen=rec["first_seen"], last_seen=rec["last_seen"],
                    occurrence=rec["occurrence"], face_image=rec.get("face_image", "")))
            results.sort(key=lambda r: r.score, reverse=True)
            return results

    def list_strangers(self, limit: int = 100, offset: int = 0) -> dict:
        with self._lock:
            total = len(self._stranger_meta)
            items = self._stranger_meta[offset: offset + limit]
            return {"total": total, "offset": offset, "limit": limit, "strangers": items}

    def clear_strangers(self) -> int:
        faiss = _get_faiss()
        with self._lock:
            count = len(self._stranger_meta)
            self._stranger_index = faiss.IndexFlatIP(self.EMBEDDING_DIM)
            self._stranger_meta  = []
            self._stranger_next_id = 1
            faces_dir = self._surv_dir / "faces"
            if faces_dir.exists():
                shutil.rmtree(faces_dir)
                faces_dir.mkdir(exist_ok=True)
            self._save_strangers()
            return count

    def get_person_face_path(self, lib_id: int, face_image: str) -> Optional[str]:
        path = self._lib_dir / f"lib_{lib_id}" / "faces" / face_image
        return str(path) if path.exists() else None
