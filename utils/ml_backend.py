"""
Shared enforcement for ML inference: optional GPU requirements for PyTorch/Ultralytics
and consistency checks with user expectations.

Ultralytics YOLO accepts both ``.pt`` and ``.onnx`` in ``YOLO_MODEL``; ONNX weights still
use the Ultralytics/ONNX stack with ``DEVICE`` for acceleration where supported.
``ML_REQUIRE_INFERENCE_GPU=1`` enforces a visible CUDA device when the configured
``DEVICE`` targets GPU (numeric index or ``cuda*`` string).
"""
from __future__ import annotations

import os
from typing import Optional


def _truthy(name: str) -> bool:
    return os.getenv(name, "").lower() in ("1", "true", "yes", "on")


def enforce_torch_cuda_for_ultralytics(component: str) -> None:
    """
    If ML_REQUIRE_INFERENCE_GPU is set, require CUDA to be available for YOLO/Ultralytics
    (device 0, cuda, etc.).
    """
    if not _truthy("ML_REQUIRE_INFERENCE_GPU"):
        return
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"[{component}] ML_REQUIRE_INFERENCE_GPU is set but torch.cuda.is_available() "
            f"is False — install CUDA PyTorch and drivers or unset ML_REQUIRE_INFERENCE_GPU."
        )


def resolve_ultralytics_device() -> str | int:
    """
    Default DEVICE from env: digit -> int CUDA index, else string (e.g. cpu, cuda:0).
    """
    raw = os.getenv("DEVICE", "0")
    if raw.isdigit():
        return int(raw)
    return raw


def require_gpu_device_if_configured(device: str | int, component: str) -> None:
    """
    If ML_REQUIRE_INFERENCE_GPU and the configured device is CUDA, require CUDA
    to be available.
    """
    if not _truthy("ML_REQUIRE_INFERENCE_GPU"):
        return
    use_cuda = False
    if isinstance(device, int):
        use_cuda = True
    elif isinstance(device, str):
        d = device.strip().lower()
        if d.isdigit():
            use_cuda = True
        elif d.startswith("cuda"):
            use_cuda = True
    if use_cuda:
        enforce_torch_cuda_for_ultralytics(component)
