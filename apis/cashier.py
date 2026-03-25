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

import json
import os
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field


#  Shared state populated by CashierService.push_result() 
_lock        = threading.Lock()
_last_result: Dict[str, Any]  = {}                                       # camera_id → latest result
_event_log  : deque           = deque(maxlen=int(os.getenv("CASHIER_LOG_MAX", "5000")))
_evidence_dir = Path(os.getenv("CASHIER_EVIDENCE_DIR", "./evidence/cashier"))

router = APIRouter()


# ─────────────────────────────────────────────
# Internal hook — called from CashierService
# ─────────────────────────────────────────────
def push_result(camera_id: str, result: Dict[str, Any]) -> None:
    """
    Called by CashierService every frame to keep the API state current.
    Logged events must contain alerts or be a transaction.
    """
    with _lock:
        _last_result[camera_id] = {**result, "camera_id": camera_id}
        if result.get("alerts") or result.get("transaction"):
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
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    try:
        full_path.resolve().relative_to(_evidence_dir.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Path traversal not allowed")
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
        raise HTTPException(status_code=500, detail=f"Could not read config: {exc}")


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

    try:
        if config_path.suffix in (".yaml", ".yml"):
            import yaml
            config_path.write_text(yaml.dump(cfg, default_flow_style=False))
        else:
            config_path.write_text(json.dumps(cfg, indent=2))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not write config: {exc}")

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
        raise HTTPException(status_code=500, detail=f"Could not write config: {exc}")

    return {"status": "reset_to_default", "config": cfg}
