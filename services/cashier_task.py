"""
services/cashier_task.py
------------------------
Cashier drawer monitor as a FrameBus task worker (refactor architecture).

FrameBus fans the same YOLO track output to every task on a channel. For
drawer/cash classes to appear in ``payload["detection"]["items"]``, configure
``YOLO_MODEL`` (and related env) to your cashier detector weights for that
deployment — do not mix COCO-only tasks on the same channel unless the model
produces compatible class indices (0=person, 1=drawer, 2=cash).

Register via POST /api/tasks with algorithmType ``CASHIER_DRAWER``.
"""

from __future__ import annotations

from typing import Any, Dict, List

from services.cashier import CashierService


class CashierDrawerTask:
    def __init__(self, task_config: dict):
        self._task_config = task_config
        self._svc = CashierService()

    def __call__(self, payload: Dict[str, Any]) -> List:
        if not self._task_config.get("enable", True):
            return []

        camera_id = str(payload.get("camera_id", ""))
        detection = payload.get("detection") or {}
        items = detection.get("items", [])

        context: Dict[str, Any] = {
            "camera_id": camera_id,
            "data": {
                "frame": payload["frame"],
                "detection": {"items": list(items), "count": len(items)},
                "use_case": {},
            },
        }
        self._svc(context)
        return []
