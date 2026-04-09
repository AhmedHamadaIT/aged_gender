"""
error_codes/response.py
-----------------------
Standardized response templates for all API endpoints.

Usage:
    from error_codes.response import success, error
    from error_codes.error_codes import ErrorCode

    # Success
    return success({"cameras": ["cam1", "cam2"], "status": "started"})

    # Error
    return error(ErrorCode.CAMERA_NOT_FOUND, detail="cam3")

Success response shape:
    {
        "status": "...",
        "error" : null,
        ...data
    }

Error response shape:
    {
        "status": "error",
        "error" : {
            "code"   : "101",
            "message": "Camera not found.",
            "detail" : "cam3"      ← optional
        }
    }
"""

from typing import Optional
from error_codes.error_codes import ErrorCode


def success(data: dict) -> dict:
    """
    Wrap a successful response.

    Args:
        data: dict of fields to include in the response

    Returns:
        {**data, "error": null}

    Example:
        return success({"status": "started", "cameras": ["cam1", "cam2"]})
    """
    return {
        **data,
        "error": None,
    }


def error(code: ErrorCode, detail: Optional[str] = None) -> dict:
    """
    Build a standardized error response.

    Args:
        code:   ErrorCode enum member
        detail: Optional extra context (e.g. the camera ID that caused the error)

    Returns:
        {
            "status": "error",
            "error": {
                "code"   : "101",
                "message": "Camera not found.",
                "detail" : "cam3"   ← only if provided
            }
        }

    Example:
        return error(ErrorCode.CAMERA_NOT_FOUND, detail="cam3")
    """
    payload = {
        "code"   : str(code.code),
        "message": code.message,
    }
    if detail:
        payload["detail"] = detail

    return {
        "status": "error",
        "error" : payload,
    }