from __future__ import annotations

import queue
import sys
import threading
import time
import types

import numpy as np


def _stub_module(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


def test_embedding_worker_skips_unavailable_backends(monkeypatch, tmp_path):
    person_calls: list[str] = []
    semantic_calls: list[str] = []

    class StubPersonSearchService:
        def __init__(self):
            self.model = None
            self.identity_manager = None

        def extract_embedding(self, crop):
            person_calls.append("called")
            raise AssertionError("person search should be skipped when the model is unavailable")

    class StubSemanticSearchService:
        def __init__(self):
            self._ready = False
            self.identity_manager = None

        def extract_embedding(self, crop):
            semantic_calls.append("called")
            raise AssertionError("semantic search should be skipped when the model is unavailable")

    monkeypatch.setitem(
        sys.modules,
        "services.person_search",
        _stub_module("services.person_search", PersonSearchService=StubPersonSearchService),
    )
    monkeypatch.setitem(
        sys.modules,
        "services.semantic_search",
        _stub_module("services.semantic_search", SemanticSearchService=StubSemanticSearchService),
    )

    import embedding_worker as worker_mod

    monkeypatch.setattr(
        worker_mod.cv2,
        "imread",
        lambda _: np.zeros((4, 4, 3), dtype=np.uint8),
    )

    embedding_queue: queue.Queue = queue.Queue()
    stop_event = threading.Event()
    errors: list[Exception] = []

    crop_path = tmp_path / "track_1.jpg"
    crop_path.write_bytes(b"not-used")
    embedding_queue.put(
        {
            "camera_id": "cam401",
            "track_id": 1,
            "crop_path": str(crop_path),
            "frame_id": 1,
            "confidence": 0.9,
            "timestamp": "2026-04-19T10:25:55Z",
        }
    )

    def _run():
        try:
            worker_mod.run_embedding_worker(embedding_queue, stop_event)
        except Exception as exc:  # pragma: no cover - assertion path captured below
            errors.append(exc)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    deadline = time.time() + 2.0
    while not embedding_queue.empty() and time.time() < deadline:
        time.sleep(0.05)

    stop_event.set()
    thread.join(timeout=2.0)

    assert not errors
    assert person_calls == []
    assert semantic_calls == []
