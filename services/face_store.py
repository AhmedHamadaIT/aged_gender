"""
services/face_store.py
----------------------
FAISS-backed face embedding store with JSON metadata sidecar.

Supports:
    - Library CRUD (create / delete / list)
    - Person CRUD (add / delete / list with multi-image registration)
    - Cross-library search (lib_ids="-1" searches all)
    - Stranger store (separate FAISS index for unknowns, auto-dedup)

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
# Face Store
# ─────────────────────────────────────────────
class FaceStore:
    """
    Thread-safe FAISS-backed face embedding store.
    Uses IndexFlatIP (inner-product on normalised vectors = cosine similarity).
    """

    EMBEDDING_DIM = 512

    def __init__(self, storage_dir: str = None):
        self._storage_dir = Path(storage_dir or os.getenv("FACE_STORAGE_DIR", "./data/face"))
        self._lib_dir     = self._storage_dir / "libraries"
        self._surv_dir    = self._storage_dir / "surveillance"
        self._lock        = threading.RLock()

        self._libraries : Dict[int, LibraryMeta] = {}
        self._indices   : Dict[int, object]      = {}
        self._emb_maps  : Dict[int, List[Tuple[int, int]]] = {}

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
                emb_map = []
                for p in raw.get("persons", []):
                    pr = PersonRecord(
                        person_id=p["person_id"], name=p["name"],
                        embedding_indices=p.get("embedding_indices", []),
                        face_images=p.get("face_images", []),
                    )
                    lib_meta.persons[pr.person_id] = pr
                    for idx in pr.embedding_indices:
                        emb_map.append((pr.person_id, idx))
                self._libraries[lib_id] = lib_meta
                self._emb_maps[lib_id]  = emb_map
                if index_path.exists():
                    self._indices[lib_id] = faiss.read_index(str(index_path))
                else:
                    self._indices[lib_id] = faiss.IndexFlatIP(self.EMBEDDING_DIM)
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
            self._indices[lib_id]   = faiss.IndexFlatIP(self.EMBEDDING_DIM)
            self._emb_maps[lib_id]  = []
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
            self._emb_maps.pop(lib_id, None)
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

            index   = self._indices[lib_id]
            emb_map = self._emb_maps[lib_id]
            start_idx   = index.ntotal
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
                index.add(emb_np)
                idx = start_idx + i
                emb_indices.append(idx)
                emb_map.append((person_id, idx))

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
            faces_dir = self._lib_dir / f"lib_{lib_id}" / "faces"
            for fname in person.face_images:
                fpath = faces_dir / fname
                if fpath.exists():
                    fpath.unlink()
            self._rebuild_index(lib_id)
            self._save_library(lib_id)
            return True

    def _rebuild_index(self, lib_id: int):
        faiss = _get_faiss()
        old_index = self._indices.get(lib_id)
        lib       = self._libraries[lib_id]
        new_index = faiss.IndexFlatIP(self.EMBEDDING_DIM)
        new_map   = []

        if old_index is not None and old_index.ntotal > 0:
            all_vecs = faiss.rev_swig_ptr(
                old_index.get_xb(), old_index.ntotal * self.EMBEDDING_DIM)
            all_vecs = np.array(all_vecs).reshape(old_index.ntotal, self.EMBEDDING_DIM)
            idx_counter = 0
            for pid, person in lib.persons.items():
                new_indices = []
                for old_idx in person.embedding_indices:
                    if old_idx < len(all_vecs):
                        vec = all_vecs[old_idx].reshape(1, -1).astype(np.float32)
                        new_index.add(vec)
                        new_indices.append(idx_counter)
                        new_map.append((pid, idx_counter))
                        idx_counter += 1
                person.embedding_indices = new_indices

        self._indices[lib_id]  = new_index
        self._emb_maps[lib_id] = new_map

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
        faiss = _get_faiss()
        with self._lock:
            query = np.array(embedding, dtype=np.float32).reshape(1, -1)
            faiss.normalize_L2(query)

            target_libs = (list(self._libraries.keys()) if lib_ids == "-1"
                           else [int(x.strip()) for x in lib_ids.split(",") if x.strip()])

            all_results = []
            for lid in target_libs:
                if lid not in self._indices:
                    continue
                index = self._indices[lid]
                if index.ntotal == 0:
                    continue
                k = min(top_k, index.ntotal)
                scores, indices = index.search(query, k)
                lib_meta = self._libraries[lid]
                emb_map  = self._emb_maps[lid]

                for i in range(k):
                    if indices[0][i] < 0:
                        continue
                    sim_score = float(scores[0][i]) * 100.0
                    if sim_score < threshold:
                        continue
                    faiss_idx = int(indices[0][i])
                    pid = None
                    for person_id, emb_idx in emb_map:
                        if emb_idx == faiss_idx:
                            pid = person_id
                            break
                    if pid is None:
                        continue
                    person = lib_meta.persons.get(pid)
                    if person is None:
                        continue
                    all_results.append(MatchResult(
                        person_id=pid, person_name=person.name, lib_id=lid,
                        score=round(sim_score, 1),
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
