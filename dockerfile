# ============================================
# YOLO Object Detection — Dockerfile
# ============================================
# Base: dustynv PyTorch for Jetson R36.4 / JetPack 6 (ARM64).
# OS/Python track the upstream tag; see hub.docker.com/r/dustynv/l4t-pytorch.
#
# This image ships with a fully working CUDA torch.
# We only add ultralytics and other non-torch deps on top.
# ============================================

FROM dustynv/l4t-pytorch:r36.4.0

# ── System dependencies ──────────────────────
# Runtime libs only. Avoid *-dev headers for GStreamer here; they often pull conflicting deps.
# GStreamer plugins are required for cv2.CAP_GSTREAMER + nvv4l2decoder on Jetson.
ENV DEBIAN_FRONTEND=noninteractive
# L4T uses ports.ubuntu.com. "invalid signature" / "not signed" during build is often:
# - corrupt apt lists or low disk on the Docker host (prune images, free space),
# - Docker engine too old for Ubuntu 22.04 gpgv inside the container (upgrade to 20.10+).
# We clear lists, retry, then fall back to insecure index fetch only if needed.
# Clear /var/cache/apt/archives/* (not only partial/): L4T bases can leave large .deb
# caches; full archives + upgrades can exhaust the build FS ("not enough free space").
# --no-upgrade avoids mass upgrades (glib/gcc toolchain) when only ffmpeg+runtime libs are needed.
RUN set -eux; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* /var/cache/apt/archives/partial/*; \
    apt-get update -o Acquire::Retries=5 -o Acquire::http::Timeout=120 \
    || apt-get update -o Acquire::Retries=5 --allow-insecure-repositories; \
    apt-get install -y --no-install-recommends --allow-unauthenticated --no-upgrade \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
        libgl1 \
        ffmpeg \
        gstreamer1.0-tools \
        gstreamer1.0-plugins-base \
        gstreamer1.0-plugins-good \
        gstreamer1.0-plugins-bad \
        gstreamer1.0-plugins-ugly \
        ; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# ── Working directory ────────────────────────
WORKDIR /app

# L4T base images may set pip's primary index to Jetson AI Lab. If that host is unreachable
# (DNS "Name or service not known"), every package (e.g. seaborn) fails. Prefer PyPI first;
# Jetson-specific wheels (ultralytics, onnxruntime-gpu) still resolve via extra index.
ENV PIP_INDEX_URL=https://pypi.org/simple \
    PIP_EXTRA_INDEX_URL=https://pypi.jetson-ai-lab.io/jp6/cu126

# Base L4T images often set pip index-url to Jetson-only mirrors; Ultralytics may try to
# `pip install lap` at runtime and fail DNS / miss wheels. Force lap from PyPI.
RUN python3 -m pip install --no-cache-dir "lap>=0.5.12" \
        --index-url https://pypi.org/simple \
        --trusted-host pypi.org \
        --trusted-host files.pythonhosted.org

# ── Find which python/pip the base image uses and install deps ──
# Use --no-deps on ultralytics to prevent pip from pulling in CPU torch
# Install all other ultralytics deps manually.
# Do NOT pip install opencv-python / opencv-python-headless: L4T base images ship OpenCV
# with GStreamer support; PyPI wheels replace it and break RTSP_BACKEND=gstreamer (CAP_GSTREAMER).
# System OpenCV is provided by the dustynv/l4t-pytorch base (or python3-opencv on some images).
RUN python3 -m pip install --no-cache-dir --no-deps \
        --index-url https://pypi.org/simple \
        --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126 \
        --trusted-host pypi.org \
        --trusted-host files.pythonhosted.org \
        --trusted-host pypi.jetson-ai-lab.io \
        ultralytics && \
    python3 -m pip install --no-cache-dir \
        --index-url https://pypi.org/simple \
        --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126 \
        --trusted-host pypi.org \
        --trusted-host files.pythonhosted.org \
        --trusted-host pypi.jetson-ai-lab.io \
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
        "uvicorn[standard]" \
        "python-multipart"\
        qdrant-client \
        gdown \
        python-multipart \
        open_clip_torch \
        redis \
        insightface \
        faiss-cpu \
        lapx

# Jetson Orin / aarch64: PyPI `onnxruntime` is often CPU-only or mismatched CUDA.
# Jetson AI Lab wheels provide CUDAExecutionProvider (+ TensorRT EP when compatible).
RUN python3 -m pip uninstall -y onnxruntime onnxruntime-gpu 2>/dev/null || true; \
    python3 -m pip install --no-cache-dir onnxruntime-gpu \
        --index-url https://pypi.org/simple \
        --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126 \
        --trusted-host pypi.org \
        --trusted-host files.pythonhosted.org \
        --trusted-host pypi.jetson-ai-lab.io

# RTSP stability for OpenCV/FFmpeg inside the container
ENV OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;tcp|timeout;5000000|reconnect;1|reconnect_delay_max;5"
ENV PYTHONUNBUFFERED=1

# ── Create directories ───────────────────────
RUN mkdir -p /app/models /app/videos /app/outputs

EXPOSE 9000
RUN mkdir -p /app/models /app/videos /app/outputs /app/data/face

# ── Default command ──────────────────────────
CMD ["sleep", "infinity"]
