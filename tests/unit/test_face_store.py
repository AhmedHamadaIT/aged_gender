"""FAISS FaceStore CRUD (requires faiss-cpu)."""

from __future__ import annotations

import numpy as np
import pytest

faiss = pytest.importorskip("faiss")

from services.face_store import FaceStore


def test_face_store_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("FACE_STORAGE_DIR", str(tmp_path))

    store = FaceStore()
    store.create_library(9001, "unit-lib")
    pid = 42
    emb = np.random.randn(512).astype("float32")
    store.add_person(9001, pid, "alice", [emb])

    matches = store.search(emb, lib_ids="9001", top_k=1, threshold=0.0)
    assert len(matches) >= 1
    assert matches[0].person_name == "alice"

    persons = store.list_persons(9001)
    assert len(persons) == 1
    assert persons[0]["name"] == "alice"

    store.delete_person(9001, pid)
    left = store.list_persons(9001)
    assert left == []
