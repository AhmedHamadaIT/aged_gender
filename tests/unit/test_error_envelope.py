"""error_codes.response helpers."""

from __future__ import annotations

from error_codes.error_codes import ErrorCode
from error_codes.response import error, success


def test_success_wraps_null_error():
    out = success({"status": "ok", "x": 1})
    assert out["error"] is None
    assert out["status"] == "ok"


def test_error_shape():
    out = error(ErrorCode.CAMERA_NOT_FOUND, detail="cam99")
    assert out["status"] == "error"
    assert out["error"]["code"] == "101"
    assert out["error"]["detail"] == "cam99"
