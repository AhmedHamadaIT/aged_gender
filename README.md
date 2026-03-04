# Age, Gender, and Mood YOLO Inference

A complete pipeline for running classification models (YOLO and MobileNetV3) on images, folders, or live webcam feeds. This project is specifically optimized to run safely on **both Desktop PCs (CPU/GPU)** and **Edge Devices like Jetson Nano/Orin**, handling PyTorch/CUDA memory issues gracefully.

## 📦 Models

This project supports three primary models:

1. **YOLO Age/Gender (`best.pt`)**: Ultralytics YOLO classification model (8 classes: Female/Male + Child/YoungAdult/MiddleAged/OldAged).
2. **GenderAge MobileNetV3 (`best_checkpoint.pth`)**: A custom MobileNetV3-Small architecture for age/gender classification.
3. **YOLO Mood (`best mood.pt`)**: Ultralytics YOLOv8 classification model for emotions (3 classes: angry, happy, neutral).

---

## 🚀 Scripts & Usage

The project provides three main scripts depending on your need:

### 1. `live_inference.py` — Real-time Multi-Model Inference
Runs **both** the YOLO Age/Gender and YOLO Mood models simultaneously on every frame. Ideal for real-time monitoring and live webcams.

```bash
# 🎥 Live Webcam (Default)
python3 live_inference.py \
    --yolo "best.pt" \
    --mood "best mood.pt" \
    --device cuda

# 📷 Single Image Test
python3 live_inference.py \
    --yolo "best.pt" \
    --mood "best mood.pt" \
    --mode image \
    --input ./images/1001.jpg \
    --device cpu

# 📁 Process a whole folder (saves annotated copies)
python3 live_inference.py \
    --yolo "best.pt" \
    --mood "best mood.pt" \
    --mode folder \
    --input ./images \
    --output ./live_output \
    --save-vis \
    --device cuda
```

### 2. `model_inference.py` — Detailed Benchmarking & Reports
Used to run extensive evaluation on a large folder of images and generate comprehensive reports (Markdown, HTML, JSON) and plots. It can run just the age/gender model, just the mood model, or both sequentially.

```bash
# Run both models and generate a combined summary
python3 model_inference.py \
    --model "best.pt" \
    --mood-model "best mood.pt" \
    --mode both \
    --input ./images \
    --output ./reports \
    --device cuda
```
*Modes available: `aged_gender` (default), `mood`, `both`.*

### 3. `compare_models.py` — Cross-Model Architecture Comparison
Used to compare the performance (FPS, memory usage, confidence) of different model architectures (YOLO vs. MobileNet) side-by-side on the same dataset.

```bash
python3 compare_models.py \
    --model-yolo "best.pt" \
    --model-gender-age "best_checkpoint.pth" \
    --model-mood "best mood.pt" \
    --images ./images \
    --output ./comparison_output \
    --num-images 100 \
    --device cuda
```

---

## ⚠️ Jetson Nano / Edge Device Considerations

This project includes specific safeguards to prevent `glibc` memory corruption and segmentation faults common on Jetson devices when mixing PyTorch, OpenCV, and Ultralytics YOLO:

1. **Import Order**: `torch` is strictly imported *before* `cv2` and `numpy`.
2. **CPU-First Loading**: YOLO `.pt` files are loaded to the CPU first, then explicitly moved to `.to("cuda")`. Loading directly to CUDA on Jetson can cause crashes.
3. **CUDA Warmup**: The scripts perform a dummy inference to force CUDA kernel compilation before real inference begins.
4. **Explicit Teardown**: `_cuda_cleanup()` and `os._exit(0)` are used to bypass Python's default garbage collection and `atexit` handlers, which notoriously double-free Unified Memory on Jetson boards.

**Conclusion:** Always pass `--device cuda` when running on a Jetson device to take advantage of the GPU safely. If an unsupported architecture is detected, the scripts will securely fall back to the CPU.
