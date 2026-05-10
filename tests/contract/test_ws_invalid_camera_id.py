"""Invalid camera_id handling for HTTP helpers."""

from __future__ import annotations

import pytest

from apis.ws_live import validate_camera_id


@pytest.mark.parametrize(
    "cid,expect_reason",
    [
        ("null", True),
        ("undefined", True),
        ("bad id!", True),
        ("cam_1", False),
        ("12", False),
    ],
)
def test_validate_camera_id(cid, expect_reason):
    r = validate_camera_id(cid)
    assert (r is not None) == expect_reason
