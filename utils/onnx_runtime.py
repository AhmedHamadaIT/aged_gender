"""
ONNX Runtime: GPU-first execution provider selection (TensorRT -> CUDA -> CPU).
Shared by all onnxruntime.InferenceSession constructors.
"""
from __future__ import annotations

import os
from typing import Any, List, Optional, Union

import onnxruntime as ort

from logger.logger_config import Logger

log = Logger.get_logger(__name__)

ProviderEntry = Union[str, tuple]


def _tensorrt_options() -> dict:
    cache = os.path.abspath(
        os.getenv("ONNX_TENSORRT_CACHE_PATH", "./trt_cache")
    )
    try:
        device_id = int(os.getenv("ONNX_TENSORRT_DEVICE_ID", "0"))
    except ValueError:
        device_id = 0
    try:
        workspace = int(os.getenv("ONNX_TENSORRT_MAX_WORKSPACE", "2147483648"))
    except ValueError:
        workspace = 2147483648
    fp16 = os.getenv("ONNX_TENSORRT_FP16", "1").lower() in ("1", "true", "yes", "on")
    cache_enable = os.getenv("ONNX_TENSORRT_ENGINE_CACHE", "1").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    return {
        "device_id": device_id,
        "trt_max_workspace_size": workspace,
        "trt_fp16_enable": fp16,
        "trt_engine_cache_enable": cache_enable,
        "trt_engine_cache_path": cache,
    }


def select_onnx_execution_providers(
    *,
    allow_tensorrt: bool = True,
) -> List[ProviderEntry]:
    """
    Return a provider list suitable for ort.InferenceSession(..., providers=...).

    Order is controlled by ``ONNX_EXECUTION_PROVIDERS_ORDER``:
    - ``tensorrt_first`` (default): TensorRT -> CUDA -> CPU
    - ``cuda_first``: CUDA -> TensorRT -> CPU (helps some Jetson ORT builds where TRT EP misbehaves)
    - ``cuda_only``: CUDA -> CPU (no TensorRT; set ``ONNX_ALLOW_TENSORRT=0`` for the same effect)
    - ``cpu_only``: CPU only (use with ``DEVICE=cpu`` / hosts without GPU ORT providers)
    """
    if os.getenv("ONNX_ALLOW_TENSORRT", "1").lower() not in (
        "1",
        "true",
        "yes",
        "on",
    ):
        allow_tensorrt = False

    order = os.getenv(
        "ONNX_EXECUTION_PROVIDERS_ORDER", "tensorrt_first"
    ).strip().lower()
    if order == "cuda_only":
        allow_tensorrt = False

    available = set(ort.get_available_providers())
    if order == "cpu_only":
        if "CPUExecutionProvider" in available:
            return ["CPUExecutionProvider"]
        log.warning(
            "[ONNX] cpu_only requested but CPUExecutionProvider missing; using fallback list"
        )
        return ["CPUExecutionProvider"]

    selected: List[ProviderEntry] = []

    trt: List[ProviderEntry] = []
    if allow_tensorrt and "TensorrtExecutionProvider" in available:
        trt.append(("TensorrtExecutionProvider", _tensorrt_options()))

    cuda: List[ProviderEntry] = []
    if "CUDAExecutionProvider" in available:
        cuda.append("CUDAExecutionProvider")

    cpu: List[ProviderEntry] = []
    if "CPUExecutionProvider" in available:
        cpu.append("CPUExecutionProvider")

    if order == "cuda_first":
        selected = cuda + trt + cpu
    else:
        selected = trt + cuda + cpu

    if not selected:
        log.warning(
            "[ONNX] No standard providers found; using CPUExecutionProvider as fallback"
        )
        return ["CPUExecutionProvider"]

    # Ensure trt cache dir exists if TensorRT is selected
    if any(
        isinstance(p, tuple) and p[0] == "TensorrtExecutionProvider"
        for p in selected
    ):
        os.makedirs(
            _tensorrt_options()["trt_engine_cache_path"],
            exist_ok=True,
        )

    return selected


def create_inference_session(
    model_path: str,
    *,
    allow_tensorrt: bool = True,
    sess_options: Optional[ort.SessionOptions] = None,
) -> ort.InferenceSession:
    """
    Create an InferenceSession with GPU-first providers and optional ORT session options.
    """
    providers = select_onnx_execution_providers(allow_tensorrt=allow_tensorrt)
    so = sess_options or ort.SessionOptions()
    session = ort.InferenceSession(
        model_path, sess_options=so, providers=providers
    )
    if _require_onnx_gpu_enforced() and not _session_uses_gpu_acceleration(session):
        raise RuntimeError(
            f"[ONNX] ML_REQUIRE_ONNX_GPU is set but no GPU/TensorRT provider is active "
            f"for {model_path!r}. Installed: {ort.get_available_providers()}"
        )
    log.info(
        "[ONNX] Session %r → active providers: %s",
        model_path,
        session.get_providers(),
    )
    return session


def _require_onnx_gpu_enforced() -> bool:
    return os.getenv("ML_REQUIRE_ONNX_GPU", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _session_uses_gpu_acceleration(session: ort.InferenceSession) -> bool:
    active = set(session.get_providers())
    return bool(
        active & {"CUDAExecutionProvider", "TensorrtExecutionProvider"}
    )


def log_session_providers(label: str, session: ort.InferenceSession) -> None:
    log.info("[ONNX] %s — active providers: %s", label, session.get_providers())
