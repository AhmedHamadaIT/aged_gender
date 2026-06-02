"""
config/settings.py
------------------
M-6: Central application settings via Pydantic BaseSettings.

All values default to the same values previously hard-coded throughout the
codebase so existing deployments need zero configuration changes.

Usage:
    from config.settings import settings

    redis_url = settings.redis_url
    model_path = settings.yolo_model
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",          # Unknown env vars are silently ignored.
    )

    # ── Redis ──────────────────────────────────────────────────────────────────
    redis_url: str = Field("redis://localhost:6379/0", alias="REDIS_URL")
    redis_password: Optional[str] = Field(None, alias="REDIS_PASSWORD")

    # ── Inference ──────────────────────────────────────────────────────────────
    yolo_model: str = Field("yolov8n.pt", alias="YOLO_MODEL")
    model_version: str = Field("unknown", alias="MODEL_VERSION")
    conf_threshold: float = Field(0.25, alias="CONF_THRESHOLD")
    iou_threshold: float = Field(0.45, alias="IOU_THRESHOLD")
    device: str = Field("cpu", alias="DEVICE")
    classes: Optional[str] = Field(None, alias="CLASSES")
    max_det: int = Field(300, alias="MAX_DET")
    imgsz: Optional[int] = Field(None, alias="IMGSZ")
    model_warmup_frames: int = Field(5, alias="MODEL_WARMUP_FRAMES")

    # ── Frame pipeline ─────────────────────────────────────────────────────────
    stream_target_fps: float = Field(15.0, alias="STREAM_TARGET_FPS")
    publish_every: int = Field(1, alias="PUBLISH_EVERY")
    task_jpeg_quality: int = Field(80, alias="TASK_JPEG_QUALITY")
    live_jpeg_quality: int = Field(75, alias="LIVE_JPEG_QUALITY")
    frame_width: int = Field(1280, alias="WIDTH")
    frame_height: int = Field(720, alias="HEIGHT")
    task_queue_maxsize: int = Field(256, alias="TASK_QUEUE_MAXSIZE")
    task_queue_highwater: float = Field(0.9, alias="TASK_QUEUE_HIGHWATER")

    # ── Tracker ───────────────────────────────────────────────────────────────
    tracker_yaml: str = Field("", alias="TRACKER_YAML")
    track_buffer_seconds: float = Field(0.0, alias="TRACK_BUFFER_SECONDS")
    botsort_track_buffer: int = Field(0, alias="BOTSORT_TRACK_BUFFER")
    botsort_track_high_thresh: float = Field(0.5, alias="BOTSORT_TRACK_HIGH_THRESH")
    botsort_track_low_thresh: float = Field(0.1, alias="BOTSORT_TRACK_LOW_THRESH")
    botsort_new_track_thresh: float = Field(0.6, alias="BOTSORT_NEW_TRACK_THRESH")

    # ── Detection filters ──────────────────────────────────────────────────────
    min_detection_area_px: int = Field(0, alias="MIN_DETECTION_AREA_PX")

    # ── Annotation ────────────────────────────────────────────────────────────
    annotation_threads: int = Field(0, alias="ANNOTATION_THREADS")
    bbox_smoothing_alpha: float = Field(0.0, alias="BBOX_SMOOTHING_ALPHA")

    # ── Shared memory ring ────────────────────────────────────────────────────
    task_shm_enabled: bool = Field(False, alias="TASK_SHM_ENABLED")
    task_shm_slot_size: int = Field(1048576, alias="TASK_SHM_SLOT_SIZE")
    task_shm_slots: int = Field(32, alias="TASK_SHM_SLOTS")

    # ── Live publish ──────────────────────────────────────────────────────────
    live_publish_require_subscriber: bool = Field(False, alias="LIVE_PUBLISH_REQUIRE_SUBSCRIBER")
    ws_backpressure_poll_sec: float = Field(1.0, alias="WS_BACKPRESSURE_POLL_SEC")
    ws_mux_enabled: bool = Field(False, alias="WS_MUX_ENABLED")
    ws_mux_queue_maxsize: int = Field(16, alias="WS_MUX_QUEUE_MAXSIZE")

    # ── WebSocket / SSE quality ────────────────────────────────────────────────
    ws_max_fps: float = Field(0.0, alias="WS_MAX_FPS")
    ws_quality_refresh_sec: float = Field(1.0, alias="WS_QUALITY_REFRESH_SEC")

    # ── Cross-line ────────────────────────────────────────────────────────────
    cross_line_debounce_sec: float = Field(0.0, alias="CROSS_LINE_DEBOUNCE_SEC")

    # ── PPE voting ────────────────────────────────────────────────────────────
    ppe_vote_window: int = Field(5, alias="PPE_VOTE_WINDOW")
    ppe_vote_threshold: float = Field(0.6, alias="PPE_VOTE_THRESHOLD")

    # ── Frozen frame ──────────────────────────────────────────────────────────
    frozen_frame_thresh: float = Field(0.9995, alias="FROZEN_FRAME_THRESH")
    frozen_frame_window: int = Field(15, alias="FROZEN_FRAME_WINDOW")
    frozen_frame_max: int = Field(10, alias="FROZEN_FRAME_MAX")

    # ── Storage ───────────────────────────────────────────────────────────────
    output_dir: str = Field("/output", alias="OUTPUT_DIR")
    save_output: bool = Field(False, alias="SAVE_OUTPUT")
    evidence_dir: str = Field("/output/evidence", alias="EVIDENCE_DIR")
    registry_restore: bool = Field(False, alias="REGISTRY_RESTORE")
    jsonl_rotate: bool = Field(False, alias="JSONL_ROTATE")
    jsonl_retain_days: int = Field(30, alias="JSONL_RETAIN_DAYS")

    # ── FAISS ─────────────────────────────────────────────────────────────────
    faiss_save_interval_sec: int = Field(300, alias="FAISS_SAVE_INTERVAL_SEC")

    # ── Observability ─────────────────────────────────────────────────────────
    log_level: str = Field("INFO", alias="LOG_LEVEL")
    perf_log_every: int = Field(100, alias="PERF_LOG_EVERY")

    # ── Security ─────────────────────────────────────────────────────────────
    api_auth_token: Optional[str] = Field(None, alias="API_AUTH_TOKEN")
    upload_max_bytes: int = Field(10 * 1024 * 1024, alias="UPLOAD_MAX_BYTES")

    # ── Prometheus ────────────────────────────────────────────────────────────
    prometheus_enabled: bool = Field(False, alias="PROMETHEUS_ENABLED")


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Return cached singleton settings instance."""
    return AppSettings()


# Module-level singleton — import this directly:
#   from config.settings import settings
settings: AppSettings = get_settings()
