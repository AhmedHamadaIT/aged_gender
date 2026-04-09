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

# ── Find which python/pip the base image uses and install deps ──
# Use --no-deps on ultralytics to prevent pip from pulling in CPU torch
# Install all other ultralytics deps manually
RUN python3 -m pip install --no-cache-dir --no-deps \
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
        "uvicorn[standard]"

# ── Create directories ───────────────────────
RUN mkdir -p /app/models /app/videos /app/outputs

# ── Default command ──────────────────────────
CMD ["sleep", "infinity"]