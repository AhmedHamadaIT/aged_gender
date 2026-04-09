"""
apis/cashier.py

FastAPI router — Cashier Monitor endpoints.

Mount in app.py:
    from apis.cashier import router as cashier_router
    app.include_router(cashier_router, prefix="/cashier", tags=["Cashier Monitor"])

Endpoints

GET    /cashier/status              — live zone state per camera
GET    /cashier/events              — paginated event log (filterable)
DELETE /cashier/events              — clear in-memory event log
GET    /cashier/evidence            — list saved evidence JPEG files
GET    /cashier/evidence/{path}     — download a single evidence JPEG
GET    /cashier/zones               — return current zone config
POST   /cashier/zones               — update zone config (no restart required)
POST   /cashier/zones/reset         — restore default centred zones
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Set

from fastapi import APIRouter, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from error_codes.error_codes import ErrorCode
from error_codes.response import error


# ─────────────────────────────────────────────
# Shared state populated by CashierService
# ─────────────────────────────────────────────
_lock         = threading.Lock()
_last_result  : Dict[str, Any] = {}   # camera_id → latest result
_event_log    : deque          = deque(maxlen=int(os.getenv("CASHIER_LOG_MAX", "5000")))
_evidence_dir  = Path(os.getenv("CASHIER_EVIDENCE_DIR", "./evidence/cashier"))


# ─────────────────────────────────────────────
# SSE Publisher — embedded, no extra file needed
# ─────────────────────────────────────────────
class _SSEPublisher:
    """
    Thread-safe SSE broadcaster.  The event loop is captured lazily the
    first time a client subscribes (which happens in an async context).
    """
    def __init__(self) -> None:
        self._queues: Dict[str, Set[asyncio.Queue]] = defaultdict(set)
        self._loop  : Optional[asyncio.AbstractEventLoop] = None

    def _ensure_loop(self) -> None:
        """Capture the running loop — must be called from async context."""
        if self._loop is None:
            try:
                self._loop = asyncio.get_event_loop()
            except RuntimeError:
                pass

    def publish(self, camera_id: str, event_type: str, payload: Dict[str, Any]) -> None:
        """Called from background worker threads."""
        if not self._loop:
            return
        msg = {"event": event_type, "camera_id": camera_id,
               "ts": time.time(), "data": payload}
        for q in list(self._queues.get(camera_id, set())):
            try:
                self._loop.call_soon_threadsafe(q.put_nowait, msg)
            except asyncio.QueueFull:
                pass

    def _subscribe(self, camera_id: str) -> "asyncio.Queue[Any]":
        self._ensure_loop()
        q: asyncio.Queue = asyncio.Queue(maxsize=200)
        self._queues[camera_id].add(q)
        return q

    def _unsubscribe(self, camera_id: str, q: "asyncio.Queue[Any]") -> None:
        self._queues[camera_id].discard(q)

    async def stream(self, camera_id: str, alert_only: bool = False) -> AsyncGenerator[str, None]:
        q = self._subscribe(camera_id)
        try:
            yield _sse("connected", {"camera_id": camera_id, "alert_only": alert_only})
            while True:
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=30.0)
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
                    continue
                if alert_only and msg["event"] == "frame":
                    continue
                yield _sse(msg["event"], msg["data"])
        except asyncio.CancelledError:
            pass
        finally:
            self._unsubscribe(camera_id, q)


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# Module-level singleton — imported by services/cashier.py
_sse_publisher = _SSEPublisher()


def sse_publish(camera_id: str, event_type: str, payload: Dict[str, Any]) -> None:
    """Public hook called directly by CashierService (no injection needed)."""
    _sse_publisher.publish(camera_id, event_type, payload)


router = APIRouter()


# ─────────────────────────────────────────────
# Internal hook — called from CashierService
# ─────────────────────────────────────────────
def push_result(camera_id: str, result: Dict[str, Any]) -> None:
    """
    Called by CashierService every frame to keep the API state current.
    Logged events must contain alerts or be a transaction.
    """
    summ = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    alerts = summ.get("alerts") or result.get("alerts") or []
    txn = bool(summ.get("transaction", result.get("transaction", False)))
    with _lock:
        _last_result[camera_id] = {**result, "camera_id": camera_id}
        if alerts or txn:
            _event_log.appendleft({
                **result,
                "camera_id": camera_id,
                "logged_at": datetime.now(timezone.utc).isoformat(),
            })


# ─────────────────────────────────────────────
# Pydantic models
# ───────────────────────────────────────────── 
class PointModel(BaseModel):
    x: float = Field(..., ge=0.0, le=1.0, description="Normalised X [0–1]")
    y: float = Field(..., ge=0.0, le=1.0, description="Normalised Y [0–1]")


class ZoneModel(BaseModel):
    shape : str              = Field("rectangle", description="'rectangle' or 'polygon'")
    points: List[PointModel] = Field(..., min_length=2)
    active: Optional[bool]   = Field(True, description="Set to false to disable this zone")


class ZoneConfigRequest(BaseModel):
    ROI_CASHIER : Optional[ZoneModel]       = None
    ROI_CUSTOMER: Optional[ZoneModel]       = None
    thresholds  : Optional[Dict[str, Any]]  = None
    detail_config: Optional[Dict[str, Any]] = Field(
        None,
        description="CASHIER_BOX_OPEN: drawerOpenLimit, serviceWaitLimit, enableStaffList, staffIds",
    )
    task: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional task_id, task_name, channel_id for integration envelope",
    )
    detection_threshold: Optional[int] = Field(
        None,
        ge=0,
        le=100,
        description="Min detection confidence 0–100 → thresholds.detection_threshold",
    )


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@router.get(
    "/status",
    summary="Live cashier status for all cameras",
    response_description="Zone counts, current case and severity per camera",
)
def get_status():
    """
    Returns the latest processed frame result for every active camera.

    Example response::

        {
          "cam1": {
            "camera_id"    : "cam1",
            "case_id"      : "N3",
            "severity"     : "NORMAL",
            "alerts"       : ["N3 EVENT: Transaction in progress"],
            "transaction"  : true,
            "cashier_zone" : {"persons": 1, "drawers": 1, "cash": 0},
            "customer_zone": {"persons": 1, "drawers": 0, "cash": 1}
          }
        }
    """
    with _lock:
        return dict(_last_result)


@router.get("/events", summary="Paginated cashier event log")
def get_events(
    severity : Optional[str] = Query(None, description="NORMAL | ALERT | CRITICAL"),
    case_id  : Optional[str] = Query(None, description="N1-N6 or A1-A7"),
    camera_id: Optional[str] = Query(None, description="Filter by camera ID"),
    limit    : int            = Query(100, ge=1, le=1000),
    offset   : int            = Query(0,   ge=0),
):
    """
    Returns logged events (alerts and transactions) newest-first.
    Use ``offset`` + ``limit`` for pagination.
    """
    with _lock:
        events = list(_event_log)

    if severity:
        events = [e for e in events if e.get("severity") == severity.upper()]
    if case_id:
        events = [e for e in events if e.get("case_id") == case_id.upper()]
    if camera_id:
        events = [e for e in events if e.get("camera_id") == camera_id]

    total = len(events)
    return {"total": total, "offset": offset, "limit": limit, "events": events[offset: offset + limit]}


@router.delete("/events", summary="Clear the in-memory event log")
def clear_events():
    with _lock:
        count = len(_event_log)
        _event_log.clear()
    return {"cleared": count}


@router.get("/evidence", summary="List saved evidence files")
def list_evidence(
    severity: Optional[str] = Query(None, description="NORMAL | ALERT | CRITICAL"),
    case_id : Optional[str] = Query(None, description="N3 / A1 / A3 …"),
    limit   : int           = Query(50, ge=1, le=500),
):
    """
    Lists saved annotated JPEG evidence files under the evidence directory.
    Each entry includes the relative path, size (KB), and modification time.
    """
    if not _evidence_dir.exists():
        return {"total": 0, "files": []}

    files = sorted(_evidence_dir.glob("**/*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)

    if severity:
        files = [f for f in files if severity.lower() in f.parts]
    if case_id:
        files = [f for f in files if case_id.upper() in f.parts]

    total = len(files)
    return {
        "total": total,
        "files": [
            {
                "path"    : str(f.relative_to(_evidence_dir)),
                "size_kb" : round(f.stat().st_size / 1024, 1),
                "modified": datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc).isoformat(),
            }
            for f in files[:limit]
        ],
    }


@router.get(
    "/evidence/{file_path:path}",
    summary="Download a specific evidence frame",
    response_class=FileResponse,
)
def download_evidence(file_path: str):
    """Download a single evidence JPEG by its relative path (from ``GET /cashier/evidence``)."""
    full_path = _evidence_dir / file_path
    if not full_path.exists() or not full_path.is_file():
        return JSONResponse(
            status_code=404,
            content=error(ErrorCode.CASHIER_EVIDENCE_NOT_FOUND, detail=file_path),
        )
    try:
        full_path.resolve().relative_to(_evidence_dir.resolve())
    except ValueError:
        return JSONResponse(
            status_code=403,
            content=error(ErrorCode.CASHIER_PATH_TRAVERSAL, detail=file_path),
        )
    return FileResponse(str(full_path), media_type="image/jpeg", filename=full_path.name)


@router.get("/zones", summary="Return current zone configuration")
def get_zones():
    """Returns the zone polygons and thresholds as stored in the config file."""
    config_path = os.getenv("CASHIER_CONFIG", "./config/cashier_zones.yaml")
    try:
        if config_path.endswith((".yaml", ".yml")):
            import yaml
            with open(config_path) as f:
                return yaml.safe_load(f) or {}
        with open(config_path) as f:
            return json.load(f)
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content=error(ErrorCode.CASHIER_CONFIG_READ_FAILED, detail=str(exc)),
        )


@router.post("/zones", summary="Update zone configuration (live, no restart needed)")
def update_zones(body: ZoneConfigRequest):
    """
    Partially or fully update zone polygons and thresholds.
    Omitted fields keep their current values. CashierService picks up the
    change on its next reload cycle (default: 60 s).

    Example body::

        {
          "ROI_CASHIER" : {"shape": "rectangle", "points": [{"x":0.0,"y":0.0},{"x":0.45,"y":1.0}]},
          "ROI_CUSTOMER": {"shape": "rectangle", "points": [{"x":0.45,"y":0.0},{"x":1.0,"y":1.0}]},
          "thresholds"  : {"drawer_open_max_seconds": 20}
        }
    """
    config_path = Path(os.getenv("CASHIER_CONFIG", "./config/cashier_zones.yaml"))
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing config
    try:
        if config_path.suffix in (".yaml", ".yml"):
            import yaml
            cfg = yaml.safe_load(config_path.read_text()) or {} if config_path.exists() else {}
        else:
            cfg = json.loads(config_path.read_text()) if config_path.exists() else {}
    except Exception:
        cfg = {}

    cfg.setdefault("zones", {})
    cfg.setdefault("thresholds", {})
    cfg.setdefault("detail_config", {})

    if body.ROI_CASHIER:
        cfg["zones"]["ROI_CASHIER"] = {
            "shape" : body.ROI_CASHIER.shape,
            "points": [[p.x, p.y] for p in body.ROI_CASHIER.points],
            "active": body.ROI_CASHIER.active,
        }
    if body.ROI_CUSTOMER:
        cfg["zones"]["ROI_CUSTOMER"] = {
            "shape" : body.ROI_CUSTOMER.shape,
            "points": [[p.x, p.y] for p in body.ROI_CUSTOMER.points],
            "active": body.ROI_CUSTOMER.active,
        }
    if body.thresholds:
        cfg["thresholds"].update(body.thresholds)
    if body.detail_config is not None:
        cfg["detail_config"] = {**cfg.get("detail_config", {}), **body.detail_config}
    if body.task is not None:
        cfg["task"] = {**cfg.get("task", {}), **body.task}
    if body.detection_threshold is not None:
        cfg["thresholds"]["detection_threshold"] = body.detection_threshold

    try:
        if config_path.suffix in (".yaml", ".yml"):
            import yaml
            config_path.write_text(yaml.dump(cfg, default_flow_style=False))
        else:
            config_path.write_text(json.dumps(cfg, indent=2))
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content=error(ErrorCode.CASHIER_CONFIG_WRITE_FAILED, detail=str(exc)),
        )

    return {"status": "updated", "config": cfg}


@router.post("/zones/reset", summary="Reset zones to default (left=cashier, right=customer)")
def reset_zones():
    """Restore built-in default zones and write them to the config file."""
    from services.cashier import _default_config

    config_path = Path(os.getenv("CASHIER_CONFIG", "./config/cashier_zones.yaml"))
    config_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = _default_config()

    try:
        if config_path.suffix in (".yaml", ".yml"):
            import yaml
            config_path.write_text(yaml.dump(cfg, default_flow_style=False))
        else:
            config_path.write_text(json.dumps(cfg, indent=2))
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content=error(ErrorCode.CASHIER_CONFIG_WRITE_FAILED, detail=str(exc)),
        )

    return {"status": "reset_to_default", "config": cfg}


# ─────────────────────────────────────────────
# SSE streaming routes
# GET /cashier/stream/{camera_id}       — all events
# GET /cashier/stream/{camera_id}/only  — alerts only
# ─────────────────────────────────────────────
_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


@router.get("/stream/{camera_id}", summary="Stream all events (SSE) for a camera")
async def stream_all(camera_id: str):
    """
    Server-Sent Events stream — every frame result for *camera_id*.
    Events: ``connected`` · ``frame`` · ``alert`` · ``gif_ready``
    """
    return StreamingResponse(
        _sse_publisher.stream(camera_id, alert_only=False),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


@router.get("/stream/{camera_id}/only", summary="Stream alerts-only SSE for a camera")
async def stream_alerts_only(camera_id: str):
    """Same as above but ``frame`` events are suppressed — only alerts and gif_ready."""
    return StreamingResponse(
        _sse_publisher.stream(camera_id, alert_only=True),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


# ─────────────────────────────────────────────
# Evidence serving routes
# ─────────────────────────────────────────────

def _find_evidence(pattern: str) -> Optional[Path]:
    """Return first match for rglob pattern under evidence dir, or None."""
    matches = list(_evidence_dir.rglob(pattern))
    return matches[0] if matches else None


@router.get("/media/{camera_id}/latest/jpg", summary="Latest evidence JPG for a camera")
async def latest_jpg(camera_id: str):
    files = [f for f in sorted(_evidence_dir.rglob(f"{camera_id}_*.jpg"))
             if "_thumb" not in f.name]
    if not files:
        return JSONResponse(
            status_code=404,
            content=error(ErrorCode.CASHIER_EVIDENCE_NOT_FOUND, detail=camera_id),
        )
    return FileResponse(str(files[-1]), media_type="image/jpeg")


@router.get("/media/{camera_id}/latest/gif", summary="Latest evidence GIF for a camera")
async def latest_gif(camera_id: str):
    files = sorted(_evidence_dir.rglob(f"{camera_id}_*.gif"))
    if not files:
        return JSONResponse(
            status_code=404,
            content=error(ErrorCode.CASHIER_EVIDENCE_NOT_FOUND, detail=camera_id),
        )
    return FileResponse(str(files[-1]), media_type="image/gif")


@router.get("/media/{camera_id}/event/{event_id}/jpg", summary="JPG for a specific event")
async def event_jpg(camera_id: str, event_id: str):
    parts = event_id.split("_")
    ts    = "_".join(parts[1:4]) if len(parts) >= 4 else event_id
    f     = _find_evidence(f"{camera_id}_{ts}*.jpg")
    if not f or "_thumb" in f.name:
        return JSONResponse(
            status_code=404,
            content=error(ErrorCode.CASHIER_EVIDENCE_NOT_FOUND, detail=event_id),
        )
    return FileResponse(str(f), media_type="image/jpeg")


@router.get("/media/{camera_id}/event/{event_id}/gif", summary="GIF for a specific event")
async def event_gif(camera_id: str, event_id: str):
    parts = event_id.split("_")
    ts    = "_".join(parts[1:4]) if len(parts) >= 4 else event_id
    f     = _find_evidence(f"{camera_id}_{ts}*.gif")
    if not f:
        return JSONResponse(
            status_code=404,
            content=error(ErrorCode.CASHIER_EVIDENCE_NOT_FOUND, detail=event_id),
        )
    return FileResponse(str(f), media_type="image/gif")


@router.get("/media/{camera_id}/drawer_count", summary="Lifetime drawer-open count from event log")
async def drawer_count_from_log(camera_id: str):
    log_path = _evidence_dir / "logs" / "events.jsonl"
    count    = 0
    if log_path.exists():
        with open(log_path) as fh:
            for line in fh:
                try:
                    rec = json.loads(line.strip())
                    if rec.get("camera_id") == camera_id and rec.get("status") == "triggered":
                        count += 1
                except json.JSONDecodeError:
                    continue
    return JSONResponse({"camera_id": camera_id, "drawer_open_count": count})

