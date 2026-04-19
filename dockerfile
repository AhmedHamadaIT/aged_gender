# ============================================
# YOLO Object Detection — Dockerfile
# ============================================
# Base: dustynv PyTorch for Jetson R36.4 / JetPack 6
# CUDA 12.8, Ubuntu 24.04, Python 3.12, ARM64
#
# This image ships with a fully working CUDA torch.
# We only add ultralytics and other non-torch deps on top.
# ============================================

FROM dustynv/l4t-pytorch:r36.4.0

# ── System dependencies ──────────────────────
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    libglib2.0-dev \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ────────────────────────
WORKDIR /app

# Base L4T images often set pip index-url to Jetson-only mirrors; Ultralytics may try to
# `pip install lap` at runtime and fail DNS / miss wheels. Force lap from PyPI.
RUN python3 -m pip install --no-cache-dir "lap>=0.5.12" \
        --index-url https://pypi.org/simple \
        --trusted-host pypi.org \
        --trusted-host files.pythonhosted.org

# ── Find which python/pip the base image uses and install deps ──
# Use --no-deps on ultralytics to prevent pip from pulling in CPU torch
# Install all other ultralytics deps manually
RUN python3 -m pip install --no-cache-dir --no-deps \
        --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 \
        --extra-index-url https://pypi.org/simple \
        ultralytics && \
    python3 -m pip install --no-cache-dir \
        --extra-index-url https://pypi.org/simple \
        opencv-python-headless \
        "numpy<2" \
        requests \
        Pillow \
        PyYAML \
        tqdm \
        python-dotenv \
        scipy \
        psutil \
        pandas \
        seaborn \
        matplotlib \
        py-cpuinfo \
        fastapi \
        onnxruntime \
        "uvicorn[standard]" \
        qdrant-client \
        gdown \
        python-multipart \
        open_clip_torch \
        redis

# RTSP stability for OpenCV/FFmpeg inside the container
ENV OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;tcp|timeout;5000000|reconnect;1|reconnect_delay_max;5"
ENV PYTHONUNBUFFERED=1
# If Ultralytics still spawns pip for optional deps, prefer PyPI as a fallback index.
ENV PIP_EXTRA_INDEX_URL=https://pypi.org/simple

# ── Create directories ───────────────────────
RUN mkdir -p /app/models /app/videos /app/outputs

EXPOSE 9000

# ── Default command ──────────────────────────
CMD ["sleep", "infinity"]