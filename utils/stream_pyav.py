"""
utils/stream_pyav.py
--------------------
Thin availability probe for the optional PyAV (av) package used by
STREAM_INGEST=pyav mode.  Importing this module is always safe — it
never raises even when PyAV is not installed.
"""
from __future__ import annotations


def pyav_available() -> bool:
    """Return True if the ``av`` (PyAV) package is importable."""
    try:
        import av  # noqa: F401

        return True
    except ImportError:
        return False
