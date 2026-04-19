"""
conftest.py — session-wide test configuration.

Stubs out heavy optional dependencies that are not installed in the test
environment (e.g. qdrant_client, torch, onnxruntime) so that tests which
import app or its sub-modules can run without a full ML environment.
"""

from __future__ import annotations

import sys
import types


def _stub_module(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


# ── qdrant_client (used by store.identity_manager) ────────────────────────────
if "qdrant_client" not in sys.modules:
    _Stub = lambda name: type(name, (), {"__init__": lambda self, *a, **kw: None})

    _qd = _stub_module("qdrant_client")
    _qd.QdrantClient = _Stub("QdrantClient")
    sys.modules["qdrant_client"] = _qd

    _qd_models = _stub_module("qdrant_client.models")
    _qd_models.PointStruct = _Stub("PointStruct")
    _qd_models.VectorParams = _Stub("VectorParams")
    _qd_models.Distance = type("Distance", (), {"COSINE": "Cosine", "EUCLID": "Euclid"})
    sys.modules["qdrant_client.models"] = _qd_models

    sys.modules["qdrant_client.http"] = _stub_module("qdrant_client.http")
    sys.modules["qdrant_client.http.models"] = _stub_module("qdrant_client.http.models")
