"""Register one task per algorithmType (API validation only)."""

from __future__ import annotations

import json

from tests.integration.helpers import cross_line_task, register_camera, register_task, unique_cam


def _phone_task(task_id: int, channel_id: str) -> dict:
    poly = [
        {
            "zone_id": "z",
            "point": [
                {"x": 0, "y": 0},
                {"x": 640, "y": 0},
                {"x": 640, "y": 480},
                {"x": 0, "y": 480},
            ],
        }
    ]
    return {
        "taskId": task_id,
        "taskName": "phone_it",
        "algorithmType": "PHONE_USAGE",
        "channelId": channel_id,
        "enable": True,
        "threshold": 50,
        "areaPosition": json.dumps(poly),
        "detailConfig": {},
        "validWeekday": ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"],
        "validStartTime": 0,
        "validEndTime": 86400000,
    }


def _mask_task(task_id: int, channel_id: str) -> dict:
    poly = [
        {
            "zone_id": "z",
            "point": [
                {"x": 0, "y": 0},
                {"x": 640, "y": 0},
                {"x": 640, "y": 480},
                {"x": 0, "y": 480},
            ],
        }
    ]
    return {
        "taskId": task_id,
        "taskName": "mask_it",
        "algorithmType": "MASK_HAIRNET_CHEF_HAT",
        "channelId": channel_id,
        "enable": True,
        "threshold": 50,
        "areaPosition": json.dumps(poly),
        "detailConfig": {"alarmType": ["no_mask"]},
        "validWeekday": ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"],
        "validStartTime": 0,
        "validEndTime": 86400000,
    }


def _cashier_task(task_id: int, channel_id: str) -> dict:
    return {
        "taskId": task_id,
        "taskName": "cashier_it",
        "algorithmType": "CASHIER_BOX_OPEN",
        "channelId": channel_id,
        "enable": True,
        "threshold": 50,
        "areaPosition": "[]",
        "detailConfig": {"drawerOpenLimit": 30, "serviceWaitLimit": 30},
        "validWeekday": ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"],
        "validStartTime": 0,
        "validEndTime": 86400000,
    }


def _face_task(task_id: int, channel_id: str) -> dict:
    return {
        "taskId": task_id,
        "taskName": "face_it",
        "algorithmType": "FACE",
        "channelId": channel_id,
        "enable": True,
        "threshold": 50,
        "areaPosition": "[]",
        "detailConfig": {"facePixelSize": 60, "qualityThreshold": 60},
        "libIds": "-1",
        "enableStranger": True,
        "validWeekday": ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"],
        "validStartTime": 0,
        "validEndTime": 86400000,
    }


def test_register_all_supported_algorithms(integration_http):
    cam = unique_cam("reg")
    register_camera(integration_http, cam)
    register_task(integration_http, cross_line_task(92001, cam, "cl"))
    register_task(integration_http, _mask_task(92002, cam))
    register_task(integration_http, _cashier_task(92003, cam))
    register_task(integration_http, _phone_task(92004, cam))
    register_task(integration_http, _face_task(92005, cam))

    r = integration_http.get("/api/tasks")
    assert r.status_code == 200
    assert r.json()["count"] >= 5
