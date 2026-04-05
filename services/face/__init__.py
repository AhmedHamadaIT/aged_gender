"""
services/face/__init__.py
--------------------------
Face recognition package.

Exports:
    FaceService  — pipeline service (register in services/__init__.py REGISTRY)
    FaceEngine   — InsightFace wrapper
    FaceStore    — FAISS-backed embedding store
"""

from .face_service import FaceService
from .face_engine  import FaceEngine
from .face_store   import FaceStore

__all__ = ["FaceService", "FaceEngine", "FaceStore"]
