"""
Utility package: ML/RTSP helpers plus re-exports from `vision_utils` (frame tools).

The top-level `vision_utils.py` module holds legacy resize/save/build_image APIs;
submodules add RTSP (`rtsp_ffmpeg`), ONNX (`onnx_runtime`), and policy checks (`ml_backend`).
"""
from __future__ import annotations

# Re-export frame / API helpers (formerly root-level `utils.py`, now `vision_utils.py`)
from vision_utils import (
    COLORS,
    annotate,
    build_image,
    draw_detections,
    draw_overlay,
    get_base_url,
    make_evidence_paths,
    resize,
    resize_for_display,
    save_frame,
)

__all__ = [
    "COLORS",
    "annotate",
    "build_image",
    "draw_detections",
    "draw_overlay",
    "get_base_url",
    "make_evidence_paths",
    "resize",
    "resize_for_display",
    "save_frame",
]
