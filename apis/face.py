"""
apis/face.py
-------------
FastAPI router — Face Recognition endpoints.

Mount in app.py:
    from apis.face import router as face_router
    app.include_router(face_router, prefix="/face", tags=["Face Recognition"])

Endpoints

  Tasks
    POST   /face/tasks               — create/update FACE task config
    GET    /face/tasks               — list configured tasks
    DELETE /face/tasks/{taskId}      — remove a task

  Libraries
    POST   /face/lib                 — create face library
    GET    /face/lib                 — list all libraries
    GET    /face/lib/{libId}         — library details + persons
    DELETE /face/lib/{libId}         — delete library

  Persons
    POST   /face/lib/{libId}/persons              — add person (multipart)
    GET    /face/lib/{libId}/persons              — list persons
    PUT    /face/lib/{libId}/persons/{personId}   — update person
    DELETE /face/lib/{libId}/persons/{personId}   — delete person

  Recognition
    POST   /face/recognize           — single-image recognition

  Attendance
    POST   /face/attendance/query    — query attendance log

  Strangers (Surveillance)
    GET    /face/strangers           — list stored strangers
    POST   /face/strangers/search   — search by face image
    DELETE /face/strangers           — clear all strangers

  Events & Evidence
    GET    /face/events              — query JSONL event log
    GET    /face/evidence/{path}     — download evidence image
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from error_codes.error_codes import ErrorCode
from error_codes.response import error, success

from logger.logger_config import Logger

log = Logger.get_logger(__name__)

router = APIRouter()

# ─────────────────────────────────────────────
# Shared service instance — set lazily
# ─────────────────────────────────────────────
_face_service = None


def _get_service():
    """Get or create the global FaceService singleton."""
    global _face_service
    if _face_service is None:
        from services.face.face_service import FaceService
        _face_service = FaceService()
    return _face_service


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
# Pydantic models
# ─────────────────────────────────────────────
class DetailConfigModel(BaseModel):
    facePixelSize         : int  = 60
    model                 : str  = "Fast"
    yawThreshold          : int  = 35
    pitchThreshold        : int  = 25
    failCount             : int  = 2
    enableAgeGenderDetect : bool = False
    enableEmotionDetect   : bool = False


class TaskCreateRequest(BaseModel):
    taskId         : int
    taskName       : str           = "attendance"
    algorithmType  : str           = "FACE"
    channelId      : int           = 0
    enable         : bool          = True
    threshold      : int           = 70
    libIds         : str           = "-1"
    enableStranger : bool          = True
    detailConfig   : DetailConfigModel = Field(default_factory=DetailConfigModel)
    validWeekday   : List[str]     = Field(default=["MONDAY","TUESDAY","WEDNESDAY","THURSDAY","FRIDAY","SATURDAY","SUNDAY"])
    validStartTime : int           = 0
    validEndTime   : int           = 86399000


class LibraryCreateRequest(BaseModel):
    libId : int
    name  : str


class AttendanceQueryRequest(BaseModel):
    personId   : Optional[int]  = None
    personName : Optional[str]  = None
    dateFrom   : Optional[str]  = None
    dateTo     : Optional[str]  = None
    limit      : int            = 100
    offset     : int            = 0


# ═════════════════════════════════════════════
# TASK ENDPOINTS
# ═════════════════════════════════════════════

@router.post("/tasks", summary="Create or update a FACE task configuration")
def create_task(req: TaskCreateRequest):
    """
    Create or update a face recognition task.

    The task defines which camera to monitor, recognition threshold,
    which libraries to match against, and quality/pose filtering params.

    Example body::

        {
          "taskId": 9,
          "taskName": "attendance",
          "threshold": 70,
          "libIds": "-1",
          "enableStranger": true,
          "detailConfig": {
            "facePixelSize": 60,
            "yawThreshold": 35,
            "pitchThreshold": 25,
            "failCount": 2
          }
        }
    """
    svc = _get_service()
    task = svc.tasks.create_task(req.model_dump())
    return success({"status": "configured", "task": task.to_dict()})


@router.get("/tasks", summary="List all configured FACE tasks")
def list_tasks():
    svc = _get_service()
    return success({"tasks": svc.tasks.list_tasks()})


@router.delete("/tasks/{task_id}", summary="Delete a FACE task")
def delete_task(task_id: int):
    svc = _get_service()
    if not svc.tasks.delete_task(task_id):
        return error(ErrorCode.FACE_TASK_NOT_FOUND, detail=str(task_id))
    return success({"status": "deleted", "taskId": task_id})


# ═════════════════════════════════════════════
# LIBRARY ENDPOINTS
# ═════════════════════════════════════════════

@router.post("/lib", summary="Create a new face library")
def create_library(req: LibraryCreateRequest):
    """
    Create a new face library for storing person embeddings.

    Example::

        {"libId": 1, "name": "Office Employees"}
    """
    svc = _get_service()
    try:
        result = svc.store.create_library(req.libId, req.name)
        return success(result)
    except ValueError as e:
        return error(ErrorCode.FACE_LIB_ALREADY_EXISTS, detail=str(e))


@router.get("/lib", summary="List all face libraries")
def list_libraries():
    svc = _get_service()
    return success({"libraries": svc.store.list_libraries()})


@router.get("/lib/{lib_id}", summary="Get library details")
def get_library(lib_id: int):
    svc = _get_service()
    result = svc.store.get_library(lib_id)
    if result is None:
        return error(ErrorCode.FACE_LIB_NOT_FOUND, detail=str(lib_id))
    return success(result)


@router.delete("/lib/{lib_id}", summary="Delete a face library")
def delete_library(lib_id: int):
    svc = _get_service()
    if not svc.store.delete_library(lib_id):
        return error(ErrorCode.FACE_LIB_NOT_FOUND, detail=str(lib_id))
    return success({"status": "deleted", "libId": lib_id})


# ═════════════════════════════════════════════
# PERSON ENDPOINTS
# ═════════════════════════════════════════════

@router.post(
    "/lib/{lib_id}/persons",
    summary="Add a person to a face library (multipart upload)",
)
async def add_person(
    lib_id    : int,
    person_id : int  = Form(...),
    name      : str  = Form(...),
    images    : List[UploadFile] = File(..., description="One or more face images"),
):
    """
    Register a new person by uploading face images.

    Send as ``multipart/form-data`` with:
        - ``person_id`` (int)
        - ``name`` (str)
        - ``images`` (one or more image files)

    The system will detect the largest face in each image, extract its
    512-d ArcFace embedding, and store it in the FAISS index.
    """
    svc = _get_service()

    if svc.store.get_library(lib_id) is None:
        return error(ErrorCode.FACE_LIB_NOT_FOUND, detail=str(lib_id))

    embeddings  = []
    face_images = []

    for img_file in images:
        img = _read_image(img_file)
        if img is None:
            return error(ErrorCode.FACE_INVALID_IMAGE, detail=img_file.filename)

        emb = svc.engine.compute_embedding_from_image(img)
        if emb is None:
            return error(ErrorCode.FACE_NO_FACE_DETECTED, detail=img_file.filename)

        embeddings.append(emb)
        face_images.append(img)

    try:
        result = svc.store.add_person(
            lib_id      = lib_id,
            person_id   = person_id,
            name        = name,
            embeddings  = embeddings,
            face_images = face_images,
        )
        return success(result)
    except ValueError as e:
        return error(ErrorCode.FACE_STORE_ERROR, detail=str(e))


@router.get("/lib/{lib_id}/persons", summary="List all persons in a library")
def list_persons(lib_id: int):
    svc = _get_service()
    persons = svc.store.list_persons(lib_id)
    if persons is None:
        return error(ErrorCode.FACE_LIB_NOT_FOUND, detail=str(lib_id))
    return success({"lib_id": lib_id, "persons": persons})


@router.put(
    "/lib/{lib_id}/persons/{person_id}",
    summary="Update a person (name and/or new face images)",
)
async def update_person(
    lib_id    : int,
    person_id : int,
    name      : Optional[str]        = Form(None),
    images    : List[UploadFile]      = File(None, description="New face images (optional)"),
):
    """
    Update an existing person. You can change the name, face images, or both.
    If new images are provided, the old embeddings are replaced.
    """
    svc = _get_service()

    if svc.store.get_library(lib_id) is None:
        return error(ErrorCode.FACE_LIB_NOT_FOUND, detail=str(lib_id))

    embeddings  = None
    face_images = None

    if images:
        embeddings  = []
        face_images = []
        for img_file in images:
            img = _read_image(img_file)
            if img is None:
                return error(ErrorCode.FACE_INVALID_IMAGE, detail=img_file.filename)
            emb = svc.engine.compute_embedding_from_image(img)
            if emb is None:
                return error(ErrorCode.FACE_NO_FACE_DETECTED, detail=img_file.filename)
            embeddings.append(emb)
            face_images.append(img)

    try:
        result = svc.store.update_person(
            lib_id      = lib_id,
            person_id   = person_id,
            name        = name,
            embeddings  = embeddings,
            face_images = face_images,
        )
        return success(result)
    except ValueError as e:
        return error(ErrorCode.FACE_PERSON_NOT_FOUND, detail=str(e))


@router.delete(
    "/lib/{lib_id}/persons/{person_id}",
    summary="Delete a person from a library",
)
def delete_person(lib_id: int, person_id: int):
    svc = _get_service()
    if not svc.store.delete_person(lib_id, person_id):
        return error(ErrorCode.FACE_PERSON_NOT_FOUND,
                     detail=f"lib={lib_id}, person={person_id}")
    return success({"status": "deleted", "lib_id": lib_id, "person_id": person_id})


# ═════════════════════════════════════════════
# RECOGNITION ENDPOINT
# ═════════════════════════════════════════════

@router.post("/recognize", summary="Single-image face recognition")
async def recognize(
    image     : UploadFile = File(..., description="Image containing one or more faces"),
    lib_ids   : str        = Form("-1", description="Library IDs (comma-separated, -1=all)"),
    threshold : int        = Form(70, description="Min match score 0-100"),
    top_k     : int        = Form(5,  description="Max results per face"),
):
    """
    Upload an image to detect and recognise all faces.

    Returns a list of detected faces, each with matching results from the
    specified libraries and face bounding boxes.
    """
    svc = _get_service()

    img = _read_image(image)
    if img is None:
        return error(ErrorCode.FACE_INVALID_IMAGE)

    faces = svc.engine.detect_and_embed(img, det_thresh=0.4)
    if not faces:
        return error(ErrorCode.FACE_NO_FACE_DETECTED)

    results = []
    for face in faces:
        face_result = {
            "face"   : face.to_dict(),
            "matches": [],
        }

        if face.embedding is not None:
            matches = svc.store.search(
                embedding = face.embedding,
                lib_ids   = lib_ids,
                top_k     = top_k,
                threshold = float(threshold),
            )
            face_result["matches"] = [
                {
                    "person_id"  : m.person_id,
                    "person_name": m.person_name,
                    "lib_id"     : m.lib_id,
                    "score"      : m.score,
                    "face_image" : m.face_image,
                }
                for m in matches
            ]

        results.append(face_result)

    return success({
        "face_count": len(results),
        "faces"     : results,
    })


# ═════════════════════════════════════════════
# ATTENDANCE ENDPOINT
# ═════════════════════════════════════════════

@router.post("/attendance/query", summary="Query attendance records")
def query_attendance(req: AttendanceQueryRequest):
    """
    Query attendance log. Returns check-in records for persons
    filtered by person ID, name, and date range.
    """
    svc = _get_service()
    result = svc.events.query_attendance(
        person_id   = req.personId,
        person_name = req.personName,
        date_from   = req.dateFrom,
        date_to     = req.dateTo,
        limit       = req.limit,
        offset      = req.offset,
    )
    return success(result)


# ═════════════════════════════════════════════
# STRANGER / SURVEILLANCE ENDPOINTS
# ═════════════════════════════════════════════

@router.get("/strangers", summary="List stored stranger faces")
def list_strangers(
    limit  : int = Query(100, ge=1, le=1000),
    offset : int = Query(0, ge=0),
):
    svc = _get_service()
    return success(svc.store.list_strangers(limit=limit, offset=offset))


@router.post("/strangers/search", summary="Search strangers by face image")
async def search_strangers(
    image     : UploadFile = File(..., description="Face image to search"),
    top_k     : int        = Form(5,  description="Max results"),
    threshold : int        = Form(30, description="Min similarity score 0-100"),
):
    """
    Upload a face image to search the stranger store.
    Returns matching stranger records with similarity scores and face images.
    """
    svc = _get_service()

    img = _read_image(image)
    if img is None:
        return error(ErrorCode.FACE_INVALID_IMAGE)

    emb = svc.engine.compute_embedding_from_image(img)
    if emb is None:
        return error(ErrorCode.FACE_NO_FACE_DETECTED)

    matches = svc.store.search_strangers(emb, top_k=top_k, threshold=float(threshold))
    return success({
        "match_count": len(matches),
        "matches"    : [
            {
                "stranger_id": m.stranger_id,
                "score"      : m.score,
                "first_seen" : m.first_seen,
                "last_seen"  : m.last_seen,
                "occurrence" : m.occurrence,
                "face_image" : m.face_image,
            }
            for m in matches
        ],
    })


@router.delete("/strangers", summary="Clear all stranger records")
def clear_strangers():
    svc = _get_service()
    count = svc.store.clear_strangers()
    return success({"status": "cleared", "count": count})


# ═════════════════════════════════════════════
# EVENTS & EVIDENCE
# ═════════════════════════════════════════════

@router.get("/events", summary="Query JSONL event log")
def get_events(
    task_id     : Optional[int]  = Query(None, description="Filter by task ID"),
    is_stranger : Optional[bool] = Query(None, description="Filter: True=strangers only, False=known only"),
    person_id   : Optional[int]  = Query(None, description="Filter by person ID"),
    limit       : int            = Query(100, ge=1, le=1000),
    offset      : int            = Query(0, ge=0),
):
    """
    Query face recognition events (newest first).
    Use filters to narrow results by task, stranger status, or person.
    """
    svc = _get_service()
    return success(svc.events.query_events(
        task_id     = task_id,
        is_stranger = is_stranger,
        person_id   = person_id,
        limit       = limit,
        offset      = offset,
    ))


@router.get(
    "/evidence/{file_path:path}",
    summary="Download an evidence image",
    response_class=FileResponse,
)
def get_evidence(file_path: str):
    """
    Download a specific evidence image by relative path.

    Valid prefixes are ``captures/``, ``faces/``, or ``scenes/``.
    """
    evidence_dir = Path(os.getenv("FACE_EVIDENCE_DIR", "./data/face/events"))
    full_path    = evidence_dir / file_path

    # Security: prevent path traversal
    try:
        full_path.resolve().relative_to(evidence_dir.resolve())
    except ValueError:
        return JSONResponse(status_code=403, content={"error": "Path traversal not allowed"})

    if not full_path.exists() or not full_path.is_file():
        return JSONResponse(status_code=404, content={"error": f"File not found: {file_path}"})

    return FileResponse(str(full_path), media_type="image/jpeg", filename=full_path.name)
