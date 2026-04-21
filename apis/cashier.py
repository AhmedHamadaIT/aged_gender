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
import uuid
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Set

from fastapi import APIRouter, HTTPException, Query
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

# Redis keys — shared across forked task workers and the API process (BUG-001/002/005/011).
_CASHIER_SSE_CHANNEL = "cashier:sse:{camera_id}"  # not live:event:* (detection stream)
_CASHIER_STATUS_HASH = "cashier:status"
_CASHIER_EVENTS_LIST = "cashier:events"

_redis_sync_lock = threading.Lock()
_redis_sync_client: Optional[Any] = None


def get_redis_sync() -> Optional[Any]:
    """
    Lazy sync Redis client for cashier IPC.
    Initialized on first use so ``REDIS_URL`` is visible after ``load_dotenv()`` (BUG-011).
    """
    global _redis_sync_client
    url = (os.getenv("REDIS_URL") or "").strip()
    if not url:
        return None
    with _redis_sync_lock:
        if _redis_sync_client is not None:
            try:
                _redis_sync_client.ping()
                return _redis_sync_client
            except Exception:
                _redis_sync_client = None
        try:
            import redis as _redis_lib

            c = _redis_lib.Redis.from_url(
                url, socket_connect_timeout=2, decode_responses=True
            )
            c.ping()
            _redis_sync_client = c
            return c
        except Exception as exc:
            print(f"[cashier] Redis lazy connect failed: {exc}")
            return None


def clear_cashier_redis_backing() -> None:
    """Clear Redis-backed cashier state (tests / admin). In-process deque is separate."""
    r = get_redis_sync()
    if r is None:
        return
    try:
        r.delete(_CASHIER_EVENTS_LIST)
        r.delete(_CASHIER_STATUS_HASH)
    except Exception:
        pass


def _redis_write_cashier_row(
    camera_id: str,
    status_snapshot: Dict[str, Any],
    log_row: Optional[Dict[str, Any]],
) -> None:
    """Persist latest status + optional log row for cross-process reads."""
    r = get_redis_sync()
    if r is None:
        return
    try:
        maxlen = int(os.getenv("CASHIER_LOG_MAX", "5000"))
        pipe = r.pipeline()
        pipe.hset(_CASHIER_STATUS_HASH, camera_id, json.dumps(status_snapshot, default=str))
        if log_row is not None:
            pipe.lpush(_CASHIER_EVENTS_LIST, json.dumps(log_row, default=str))
            pipe.ltrim(_CASHIER_EVENTS_LIST, 0, max(0, maxlen - 1))
        pipe.execute()
    except Exception:
        pass


# ─────────────────────────────────────────────
# SSE Publisher — embedded, no extra file needed
# ─────────────────────────────────────────────
class _SSEPublisher:
    """
    Thread-safe SSE broadcaster with dual delivery:

    Path 1 — asyncio queues (in-process):
        The worker thread calls publish() which schedules onto the event loop.
        Works in a single uvicorn worker. Always active.

    Path 2 — Redis Pub/Sub (multi-worker):
        publish() also does redis.publish() synchronously on the worker thread.
        stream() subscribes to Redis so any uvicorn worker can serve cashier SSE.
        Active only when REDIS_URL is set and Redis is reachable.
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
        msg = {"event": event_type, "camera_id": camera_id,
               "ts": time.time(), "data": payload}

        # Path 1: in-process asyncio queues
        if self._loop:
            for q in list(self._queues.get(camera_id, set())):
                try:
                    self._loop.call_soon_threadsafe(q.put_nowait, msg)
                except asyncio.QueueFull:
                    pass

        # Path 2: Redis pub/sub on dedicated channel (fork-safe; not live:event:*)
        r = get_redis_sync()
        if r is not None:
            try:
                import json as _json

                r.publish(_CASHIER_SSE_CHANNEL.format(camera_id=camera_id), _json.dumps(msg))
            except Exception:
                pass  # never block inference on Redis errors

    def _subscribe(self, camera_id: str) -> "asyncio.Queue[Any]":
        self._ensure_loop()
        q: asyncio.Queue = asyncio.Queue(maxsize=200)
        self._queues[camera_id].add(q)
        return q

    def _unsubscribe(self, camera_id: str, q: "asyncio.Queue[Any]") -> None:
        self._queues[camera_id].discard(q)

    async def stream(self, camera_id: str, alert_only: bool = False) -> AsyncGenerator[str, None]:
        """
        Yields SSE strings.  Delivers from both the in-process asyncio queue
        and (when available) a Redis Pub/Sub subscription so that this method
        works across multiple uvicorn workers.
        """
        self._ensure_loop()
        q = self._subscribe(camera_id)

        # Start Redis subscriber coroutine that feeds into the same asyncio queue
        redis_task: Optional[asyncio.Task] = None
        redis_url = os.getenv("REDIS_URL", "")
        if redis_url:
            redis_task = asyncio.create_task(
                self._redis_to_queue(camera_id, q), name=f"cashier_redis_{camera_id}"
            )

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
            if redis_task is not None:
                redis_task.cancel()
                try:
                    await redis_task
                except asyncio.CancelledError:
                    pass

    async def _redis_to_queue(
        self, camera_id: str, q: "asyncio.Queue[Any]"
    ) -> None:
        """Subscribe to Redis cashier:sse:{camera_id} and forward into *q*."""
        redis_url = os.getenv("REDIS_URL", "")
        try:
            import redis.asyncio as aioredis
            import json as _json
        except ImportError:
            return

        client = None
        try:
            client = aioredis.from_url(redis_url, socket_connect_timeout=2)
            await client.ping()
            async with client.pubsub() as ps:
                await ps.subscribe(_CASHIER_SSE_CHANNEL.format(camera_id=camera_id))
                async for raw_msg in ps.listen():
                    if raw_msg["type"] != "message":
                        continue
                    raw = raw_msg["data"]
                    try:
                        msg = _json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
                    except Exception:
                        continue
                    try:
                        q.put_nowait(msg)
                    except asyncio.QueueFull:
                        pass
        except asyncio.CancelledError:
            pass
        except Exception:
            pass
        finally:
            if client is not None:
                try:
                    await client.aclose()
                except Exception:
                    pass


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# Module-level singleton — imported by services/cashier.py
_sse_publisher = _SSEPublisher()


def sse_publish(camera_id: str, event_type: str, payload: Dict[str, Any]) -> None:
    """Public hook called from ``CashierDrawerTask`` (worker thread) for ``/cashier/stream``."""
    _sse_publisher.publish(camera_id, event_type, payload)


router = APIRouter()


# ─────────────────────────────────────────────
# Internal hooks — called from CashierDrawerTask / tests
# ─────────────────────────────────────────────
_CASHIER_EVENT_TYPE = "CASHIER_BOX_OPEN"


def push_structured_cashier_event(camera_id: str, event: Dict[str, Any]) -> None:
    """
    Canonical cashier output: one structured event per processed frame.
    Updates live status and appends to the in-memory event log (newest-first).
    When Redis is configured, mirrors to ``cashier:status`` / ``cashier:events`` for forked workers.
    """
    logged = {
        **event,
        "camera_id": camera_id,
        "logged_at": datetime.now(timezone.utc).isoformat(),
    }
    snap = {**event, "camera_id": camera_id}
    with _lock:
        _last_result[camera_id] = snap
        _event_log.appendleft(logged)
    _redis_write_cashier_row(camera_id, snap, logged)


def _legacy_hybrid_to_structured(camera_id: str, result: Dict[str, Any]) -> Dict[str, Any]:
    """Build a CASHIER_BOX_OPEN structured event from pre-refactor test payloads."""
    summ = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    case_id = result.get("case_id") or summ.get("case_id", "N1")
    severity = result.get("severity") or summ.get("severity", "NORMAL")
    cz = summ.get("cashier_zone") if isinstance(summ.get("cashier_zone"), dict) else {}
    kz = summ.get("customer_zone") if isinstance(summ.get("customer_zone"), dict) else {}

    def _lvl(sev: str) -> str:
        if sev == "NORMAL":
            return "INFO"
        if sev == "ALERT":
            return "WARNING"
        return "CRITICAL"

    ps_obj = {
        "case_matched": case_id,
        "case_level": _lvl(severity),
        "alert_triggered": severity != "NORMAL",
        "critical_triggered": severity == "CRITICAL",
        "total_open_count": 0,
        "total_open_duration_ms": 0,
        "current_open_duration_ms": 0,
        "zones": {
            "cashier": {
                "persons_count": int(cz.get("persons", 0)),
                "drawers_count": int(cz.get("drawers", 0)),
                "cash_count": int(cz.get("cash", 0)),
                "unauthorized_present": False,
            },
            "customer": {
                "persons_count": int(kz.get("persons", 0)),
                "cash_count": int(kz.get("cash", 0)),
            },
        },
        "detections": [],
        "drawer_open_duration_ms": 0,
    }
    now_ms = int(time.time() * 1000)
    date_utc = (
        datetime.fromtimestamp(now_ms / 1000, tz=timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
        + "Z"
    )
    data = {
        "algorithmType": _CASHIER_EVENT_TYPE,
        "captureId": f"{_CASHIER_EVENT_TYPE}_{uuid.uuid4()}.jpg",
        "sceneId": f"{_CASHIER_EVENT_TYPE}_{uuid.uuid4()}.jpg",
        "channelId": str(camera_id) if camera_id else "",
        "channelName": "",
        "deviceSN": "UNKNOWN",
        "id": uuid.uuid4().hex,
        "taskId": None,
        "taskName": "",
        "recordTime": now_ms,
        "dateUTC": date_utc,
        "total_open_count": 0,
        "total_open_duration_ms": 0,
        "current_open_duration_ms": 0,
        "personStructural": json.dumps(ps_obj, separators=(",", ":")),
        "captureUrl": "",
        "sceneUrl": "",
    }
    return {
        "eventId": uuid.uuid4().hex,
        "eventType": _CASHIER_EVENT_TYPE,
        "timestamp": now_ms,
        "timestampUTC": date_utc,
        "taskId": None,
        "taskName": "",
        "channelId": str(camera_id) if camera_id else "",
        "camera_id": camera_id,
        "case_id": case_id,
        "severity": severity,
        "data": data,
    }


def push_result(camera_id: str, result: Dict[str, Any]) -> None:
    """
    Back-compat for tests and older callers.

    If ``result`` is already a structured cashier event (``eventType`` + ``data``),
    behaves like :func:`push_structured_cashier_event`.
    Otherwise coerces legacy hybrid dicts into the structured shape; status is
    always updated, but the event log only grows when there are alerts or a
    transaction (previous behavior).
    """
    if result.get("eventType") == _CASHIER_EVENT_TYPE and isinstance(result.get("data"), dict):
        push_structured_cashier_event(camera_id, result)
        return

    summ = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    alerts = summ.get("alerts") or result.get("alerts") or []
    txn = bool(summ.get("transaction", result.get("transaction", False)))
    event = _legacy_hybrid_to_structured(camera_id, result)
    snap = {**event, "camera_id": camera_id}
    with _lock:
        _last_result[camera_id] = snap
        if alerts or txn:
            logged = {
                **event,
                "camera_id": camera_id,
                "logged_at": datetime.now(timezone.utc).isoformat(),
            }
            _event_log.appendleft(logged)
            _redis_write_cashier_row(camera_id, snap, logged)
        else:
            _redis_write_cashier_row(camera_id, snap, None)


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
    response_description="Latest structured cashier event per camera (see data / case_id / severity)",
)
def get_status():
    """
    Returns the latest structured cashier event per camera (same object as
    ``GET /detection/stream`` bodies for ``CASHIER_BOX_OPEN``): top-level
    ``eventType``, ``taskId``, ``case_id``, ``severity``, Eyego ``data`` block, etc.

    Example (fields abbreviated)::

        {
          "cam1": {
            "camera_id" : "cam1",
            "eventType" : "CASHIER_BOX_OPEN",
            "case_id"   : "N3",
            "severity"  : "NORMAL",
            "data"      : { "algorithmType": "CASHIER_BOX_OPEN", "personStructural": "..." }
          }
        }
    """
    r = get_redis_sync()
    if r is not None:
        try:
            raw = r.hgetall(_CASHIER_STATUS_HASH)
            if raw:
                out: Dict[str, Any] = {}
                for k, v in raw.items():
                    try:
                        out[k] = json.loads(v)
                    except (TypeError, json.JSONDecodeError):
                        continue
                if out:
                    return out
        except Exception:
            pass
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
    Returns structured cashier events (one per processed frame when the task worker
    is running) newest-first. Legacy :func:`push_result` test payloads still append
    only on alerts/transaction.
    Use ``offset`` + ``limit`` for pagination.
    """
    events: List[Dict[str, Any]] = []
    r = get_redis_sync()
    if r is not None:
        try:
            if int(r.llen(_CASHIER_EVENTS_LIST) or 0) > 0:
                for row in r.lrange(_CASHIER_EVENTS_LIST, 0, -1):
                    try:
                        events.append(json.loads(row))
                    except (TypeError, json.JSONDecodeError):
                        continue
        except Exception:
            events = []
    if not events:
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
    clear_cashier_redis_backing()
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
    try:
        base = _evidence_dir.resolve()
        target = (base / file_path).resolve()
        target.relative_to(base)
    except (ValueError, OSError):
        return JSONResponse(
            status_code=403,
            content=error(ErrorCode.CASHIER_PATH_TRAVERSAL, detail=file_path),
        )
    if not target.is_file():
        return JSONResponse(
            status_code=404,
            content=error(ErrorCode.CASHIER_EVIDENCE_NOT_FOUND, detail=file_path),
        )
    return FileResponse(str(target), media_type="image/jpeg", filename=target.name)


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
        raise HTTPException(
            status_code=404,
            detail=f"Cashier evidence not found for camera {camera_id!r}.",
        )
    return FileResponse(str(files[-1]), media_type="image/jpeg")


@router.get("/media/{camera_id}/latest/gif", summary="Latest evidence GIF for a camera")
async def latest_gif(camera_id: str):
    files = sorted(_evidence_dir.rglob(f"{camera_id}_*.gif"))
    if not files:
        raise HTTPException(
            status_code=404,
            detail=f"Cashier evidence GIF not found for camera {camera_id!r}.",
        )
    return FileResponse(str(files[-1]), media_type="image/gif")


@router.get("/media/{camera_id}/event/{event_id}/jpg", summary="JPG for a specific event")
async def event_jpg(camera_id: str, event_id: str):
    parts = event_id.split("_")
    ts    = "_".join(parts[1:4]) if len(parts) >= 4 else event_id
    f     = _find_evidence(f"{camera_id}_{ts}*.jpg")
    if not f or "_thumb" in f.name:
        raise HTTPException(
            status_code=404,
            detail=f"Cashier evidence JPEG not found for event {event_id!r}.",
        )
    return FileResponse(str(f), media_type="image/jpeg")


@router.get("/media/{camera_id}/event/{event_id}/gif", summary="GIF for a specific event")
async def event_gif(camera_id: str, event_id: str):
    parts = event_id.split("_")
    ts    = "_".join(parts[1:4]) if len(parts) >= 4 else event_id
    f     = _find_evidence(f"{camera_id}_{ts}*.gif")
    if not f:
        raise HTTPException(
            status_code=404,
            detail=f"Cashier evidence GIF not found for event {event_id!r}.",
        )
    return FileResponse(str(f), media_type="image/gif")


@router.get(
    "/media/{camera_id}/drawer_count",
    summary="Lifetime drawer-open edge count (cashier ROI) from persisted totals",
)
async def drawer_count_from_log(camera_id: str):
    """
    Reads ``cashier_drawer_open_totals.json`` under the evidence ``logs/`` folder
    (same file ``CashierService`` maintains). This is cumulative closed→open edges
    per camera, not a tally of legacy ``events.jsonl`` trigger rows.
    """
    totals_path = _evidence_dir / "logs" / "cashier_drawer_open_totals.json"
    count = 0
    if totals_path.is_file():
        try:
            raw = json.loads(totals_path.read_text(encoding="utf-8"))
            m = raw.get("by_camera")
            if isinstance(m, dict) and camera_id in m:
                count = int(m[camera_id])
        except (json.JSONDecodeError, TypeError, ValueError):
            count = 0
    return JSONResponse({"camera_id": camera_id, "drawer_open_count": count})

