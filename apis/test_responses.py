
"""
apis/test_responses.py
----------------------
Test endpoint to preview all success and error response shapes.
Remove or disable in production.

Endpoint registered in app.py:
    GET /test/responses → returns all possible response shapes
"""

from error_codes.error_codes import ErrorCode
from error_codes.response import success, error


def get_test_responses() -> dict:
    return {

        # ── Success examples ──────────────────
        "success_examples": {

            "camera_add": success({
                "status" : "configured",
                "cameras": {
                    "cam1": "rtsp://admin:pass@192.168.1.10:554/Streaming/channels/1101",
                    "cam2": "rtsp://admin:pass@192.168.1.10:554/Streaming/channels/1401",
                },
            }),

            "camera_list": success({
                "count"  : 2,
                "cameras": [
                    {"id": "cam1", "url": "rtsp://..."},
                    {"id": "cam2", "url": "rtsp://..."},
                ],
            }),

            "camera_delete": success({
                "status"   : "removed",
                "camera_id": "cam1",
                "remaining": ["cam2"],
            }),

            "detection_setup": success({
                "status"  : "configured",
                "pipeline": ["detector", "age_gender"],
            }),

            "detection_start": success({
                "status" : "started",
                "cameras": ["cam1", "cam2"],
            }),

            "detection_stop": success({
                "status" : "stopped",
                "cameras": ["cam1", "cam2"],
            }),
        },

        # ── Error examples ────────────────────
        "error_examples": {

            # 1xx Camera
            "101_camera_not_found": error(
                ErrorCode.CAMERA_NOT_FOUND, detail="cam3"
            ),
            "102_camera_already_running": error(
                ErrorCode.CAMERA_ALREADY_RUNNING, detail="cam1"
            ),
            "103_no_cameras_configured": error(
                ErrorCode.NO_CAMERAS_CONFIGURED
            ),

            # 2xx Detection
            "201_pipeline_not_configured": error(
                ErrorCode.PIPELINE_NOT_CONFIGURED
            ),
            "202_unknown_service": error(
                ErrorCode.UNKNOWN_SERVICE, detail="['fake_service']"
            ),
            "203_no_cameras_running": error(
                ErrorCode.NO_CAMERAS_RUNNING
            ),
            "204_camera_process_error": error(
                ErrorCode.CAMERA_PROCESS_ERROR, detail="cam1"
            ),

            # 3xx Stream
            "301_stream_camera_not_found": error(
                ErrorCode.STREAM_CAMERA_NOT_FOUND, detail="cam3"
            ),

            # 5xx System
            "501_pipeline_not_set": error(
                ErrorCode.PIPELINE_NOT_SET
            ),
            "502_internal_error": error(
                ErrorCode.INTERNAL_ERROR, detail="Unexpected exception."
            ),
        },

        # ── Error code reference ──────────────
        "error_code_reference": {
            e.code: e.message for e in ErrorCode
        },
    }