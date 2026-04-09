"""
error_codes/error_codes.py
--------------------------
Centralized error code registry.

Ranges:
    1xx — Camera errors
    2xx — Detection errors
    3xx — Stream errors
    4xx — Cashier errors
    5xx — System errors
"""

from enum import Enum


class ErrorCode(Enum):

    # 1xx Camera
    CAMERA_NOT_FOUND        = 101, "Camera not found."
    CAMERA_ALREADY_RUNNING  = 102, "Camera is already running."
    NO_CAMERAS_CONFIGURED   = 103, "No cameras configured. Call POST /cameras first."

    # 2xx Detection
    PIPELINE_NOT_CONFIGURED = 201, "Pipeline not configured. Call POST /detection/setup first."
    UNKNOWN_SERVICE         = 202, "Unknown service in pipeline."
    NO_CAMERAS_RUNNING      = 203, "No cameras are currently running."
    CAMERA_PROCESS_ERROR    = 204, "Camera process encountered an error."

    # 3xx Stream
    STREAM_CAMERA_NOT_FOUND = 301, "Stream not found for the requested camera."

    # 4xx Cashier
    CASHIER_CONFIG_READ_FAILED  = 401, "Could not read cashier config."
    CASHIER_CONFIG_WRITE_FAILED = 402, "Could not write cashier config."
    CASHIER_PATH_TRAVERSAL       = 403, "Path traversal not allowed."
    CASHIER_EVIDENCE_NOT_FOUND   = 404, "Cashier evidence not found."

    # 5xx System
    PIPELINE_NOT_SET        = 501, "Pipeline factory not set."
    INTERNAL_ERROR          = 502, "Internal server error."

    def __init__(self, code: int, message: str):
        self.code    = code
        self.message = message