"""
apis/face_lib.py
-----------------
Face library management routes — all endpoints defined here as an APIRouter.

Endpoints:
    POST   /api/face/lib                         → create library
    GET    /api/face/lib                         → list libraries
    GET    /api/face/lib/{libId}                 → library details
    DELETE /api/face/lib/{libId}                 → delete library

    POST   /api/face/lib/{libId}/persons         → add person (multipart)
    GET    /api/face/lib/{libId}/persons         → list persons
    DELETE /api/face/lib/{libId}/persons/{pid}   → delete person

    POST   /api/face/recognize                   → single-image recognition
    POST   /api/face/strangers/search            → search strangers by face
    GET    /api/face/strangers                   → list strangers
    DELETE /api/face/strangers                   → clear all strangers
"""

from typing import List, Optional

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query
from utils.auth import check_upload_size, require_auth


router = APIRouter(prefix="/api/face", tags=["face"])

# ─────────────────────────────────────────────
# Shared instances — initialised lazily
# ─────────────────────────────────────────────
_engine = None
_store  = None


def _get_engine():
    global _engine
    if _engine is None:
        from services.face_engine import FaceEngine
        _engine = FaceEngine()
    return _engine


def _get_store():
    global _store
    if _store is None:
        from services.face_store import FaceStore
        _store = FaceStore()
    return _store


def _read_image(upload: UploadFile) -> Optional[np.ndarray]:
    """Read an uploaded file into a BGR numpy array."""
    try:
        data = upload.file.read()
        arr  = np.frombuffer(data, dtype=np.uint8)
        img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


# ─────────────────────────────────────────────
# Library routes
# ─────────────────────────────────────────────
@router.post("/lib")
def create_lib(lib_id: int = Form(...), name: str = Form(...)):
    store = _get_store()
    try:
        result = store.create_library(lib_id, name)
        return {"status": "created", "library": result}
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.get("/lib")
def list_libs():
    return {"libraries": _get_store().list_libraries()}


@router.get("/lib/{lib_id}")
def get_lib(lib_id: int):
    result = _get_store().get_library(lib_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Library {lib_id} not found.")
    return result


@router.delete("/lib/{lib_id}")
def delete_lib(lib_id: int):
    if not _get_store().delete_library(lib_id):
        raise HTTPException(status_code=404, detail=f"Library {lib_id} not found.")
    return {"status": "deleted", "libId": lib_id}


# ─────────────────────────────────────────────
# Person routes
# ─────────────────────────────────────────────
@router.post("/lib/{lib_id}/persons", dependencies=[Depends(require_auth), Depends(check_upload_size)])
async def add_person(
    lib_id   : int,
    person_id: int              = Form(...),
    name     : str              = Form(...),
    images   : List[UploadFile] = File(...),
):
    engine = _get_engine()
    store  = _get_store()

    if store.get_library(lib_id) is None:
        raise HTTPException(status_code=404, detail=f"Library {lib_id} not found.")

    embeddings  = []
    face_images = []

    import asyncio as _aio

    for img_file in images:
        img = _read_image(img_file)
        if img is None:
            raise HTTPException(status_code=400, detail=f"Cannot read image: {img_file.filename}")

        # M-13: run embedding in a thread to avoid blocking the event loop.
        emb = await _aio.to_thread(engine.embedding_from_image, img)
        if emb is None:
            raise HTTPException(
                status_code=400,
                detail=f"No face detected in: {img_file.filename}",
            )
        embeddings.append(emb)
        face_images.append(img)

    try:
        result = store.add_person(
            lib_id=lib_id, person_id=person_id, name=name,
            embeddings=embeddings, face_images=face_images,
        )
        return {"status": "created", "person": result}
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.get("/lib/{lib_id}/persons")
def list_persons(lib_id: int):
    store   = _get_store()
    persons = store.list_persons(lib_id)
    if persons is None:
        raise HTTPException(status_code=404, detail=f"Library {lib_id} not found.")
    return {"lib_id": lib_id, "persons": persons}


@router.delete("/lib/{lib_id}/persons/{person_id}")
def delete_person(lib_id: int, person_id: int):
    if not _get_store().delete_person(lib_id, person_id):
        raise HTTPException(
            status_code=404,
            detail=f"Person {person_id} not found in library {lib_id}.",
        )
    return {"status": "deleted", "lib_id": lib_id, "person_id": person_id}


# ─────────────────────────────────────────────
# Recognition route
# ─────────────────────────────────────────────
@router.post("/recognize", dependencies=[Depends(check_upload_size)])
async def recognize(
    image    : UploadFile = File(...),
    lib_ids  : str        = Form("-1"),
    threshold: int        = Form(70),
    top_k    : int        = Form(5),
):
    engine = _get_engine()
    store  = _get_store()

    img = _read_image(image)
    if img is None:
        raise HTTPException(status_code=400, detail="Cannot read image.")

    import asyncio as _aio
    faces = await _aio.to_thread(engine.detect_and_embed, img, 0.4)
    if not faces:
        raise HTTPException(status_code=400, detail="No face detected in image.")

    results = []
    for face in faces:
        face_result = {"face": face.to_dict(), "matches": []}
        if face.embedding is not None:
            matches = store.search(
                embedding=face.embedding, lib_ids=lib_ids,
                top_k=top_k, threshold=float(threshold),
            )
            face_result["matches"] = [
                {"person_id": m.person_id, "person_name": m.person_name,
                 "lib_id": m.lib_id, "score": m.score, "face_image": m.face_image}
                for m in matches
            ]
        results.append(face_result)

    return {"face_count": len(results), "faces": results}


# ─────────────────────────────────────────────
# Stranger routes
# ─────────────────────────────────────────────
@router.get("/strangers")
def list_strangers(
    limit : int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    return _get_store().list_strangers(limit=limit, offset=offset)


@router.post("/strangers/search", dependencies=[Depends(check_upload_size)])
async def search_strangers(
    image    : UploadFile = File(...),
    top_k    : int        = Form(5),
    threshold: int        = Form(30),
):
    engine = _get_engine()
    store  = _get_store()

    img = _read_image(image)
    if img is None:
        raise HTTPException(status_code=400, detail="Cannot read image.")

    emb = engine.embedding_from_image(img)
    if emb is None:
        raise HTTPException(status_code=400, detail="No face detected in image.")

    matches = store.search_strangers(emb, top_k=top_k, threshold=float(threshold))
    return {
        "match_count": len(matches),
        "matches": [
            {"stranger_id": m.stranger_id, "score": m.score,
             "first_seen": m.first_seen, "last_seen": m.last_seen,
             "occurrence": m.occurrence, "face_image": m.face_image}
            for m in matches
        ],
    }


@router.delete("/strangers")
def clear_strangers():
    count = _get_store().clear_strangers()
    return {"status": "cleared", "count": count}
